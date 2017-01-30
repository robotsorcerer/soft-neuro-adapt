--[[ 	
		This code publishes the controller messages 
		to myrio which in turn controls the valve circuit boards

		Author: Olalekan Ogunmolu
		Date: Nov 9, 2016
		]]

require 'torch'
require 'nn'
require 'rnn'
require 'nngraph'
require 'optim'  --for graphing our loss

local ros = require 'ros'
local json = require 'cjson'
local c = require 'trepl.colorize'

-- for memory optimizations and graph generation
local optnet = require 'optnet'
local graphgen = require 'optnet.graphgen'
local iterm = require 'iterm'
require 'iterm.dot'

--options and general settings -----------------------------------------------------
local cmd = torch.CmdLine()
cmd:option('-model', 'lstm', 'mlp or lstm')
cmd:option('-hiddenSize', {1,10,1}, 'hidden size')
cmd:option('-outputSize', 1, 'output size')
cmd:option('-rho', 5, 'BPTT steps')
cmd:option('-lr', 1e-4, 'learning rate')
cmd:option('-seed', 123, 'manual seed')
cmd:option('rate', 1e-2, 'ros sleep rate in seconds')

cmd:option('save', 'logs', 'save logs')
cmd:option('graph', false, 'generate net graph')
cmd:option('netdir', 'network', 'network path')

--dropout
cmd:option('-dropout', true, 'apply dropout with this probability after each rnn layer. dropout <= 0 disables it.')
cmd:option('-dropoutProb', 0.35, 'probability of zeroing a neuron (dropout probability)')

--Gpu settings
cmd:option('-gpu', 0, 'which gpu to use. -1 = use CPU; >=0 use gpu')
cmd:option('-backend', 'cudnn', 'nn|cudnn')

opt = cmd:parse(arg or {})
torch.manualSeed(opt.seed)
torch.setnumthreads(8)

for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

if opt.gpu == -1 then torch.setdefaulttensortype('torch.DoubleTensor') end

use_cuda = false

if opt.gpu >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpu + 1)                         -- +1 because lua is 1-indexed
  idx       = cutorch.getDevice()
  print('\nSystem has', cutorch.getDeviceCount(), 'gpu(s).', 'Code is running on GPU:', idx)
  use_cuda = true  
end

ros.init('lmfc_controller')

spinner = ros.AsyncSpinner()
spinner:start()

local nh = ros.NodeHandle()
local lmfcClient = nh:serviceClient('/error_srv', 'nn_controller/amfcError', true)
local ok = lmfcClient:exists()
local timeout = ros.Duration(5)
local ok = lmfcClient:waitForExistence(timeout)

print('Calling service: ' .. lmfcClient:getService())
local sc = ros.ServiceClient('/error_srv', 'nn_controller/amfcError', true)

local function net_controller()
	local net_controller = nn.Sequential()
	if opt.model == 'mlp' then 
		net_controller:add(nn.Linear(1, 3))
		net_controller:add(nn.ReLU())
		net_controller:add(nn.Linear(3, 1))
	elseif opt.model == 'lstm' then
		nn.FastLSTM.usenngraph = false -- faster
		local hidden = opt.hiddenSize
		local inputSize = hidden[1]
		local rnn
		for i, hiddenSize in ipairs(hidden) do 
			rnn = nn.FastLSTM(inputSize, hiddenSize, opt.rho)
			net_controller:add(rnn)
		    inputSize = hiddenSize
		end
		net_controller = nn.Sequencer(net_controller, opt.outputSize)
	end
	local cost = nn.MSECriterion()
	return cost, net_controller
end


local params = lmfcClient:call()
local am, km, ref, target = 	params.am, params.km, params.ref, params.y

local function get_target()
	local lmfcClient = nh:serviceClient('/error_srv', 'nn_controller/amfcError', true)
	local ok = lmfcClient:exists()
	local target, err, input, param

	if ok then 
		param = lmfcClient:call()
		target, err = param.y, param.error
	end
	local y = torch.DoubleTensor(1):fill(target); 
	local e = torch.DoubleTensor(1):fill(err);	
	return param, y, e, ok
end

local function saveNet(net)
	--check if network directory exists
	if not paths.dirp(opt.netdir)  then
	  paths.mkdir(opt.netdir)
	end

	local netname = opt.model .. '_net.t7'

	local filename = paths.concat(opt.netdir, netname)
	if net.iter == 0 then
	  if paths.filep(filename) then
	   os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
	  end  
	  os.execute('mkdir -p ' .. sys.dirname(filename))
	else    
	  print('<trainer> saving network model to '..filename)
	  torch.save(filename, net.network)
	end
end

local function move2GPU(x)
	if use_cuda then
		x:cuda()
	else
		x:double()
	end
	return x
end

print(c.blue '==>' ..' configuring model')
cost, neunet = net_controller()

cost 	= move2GPU(cost)
neunet 	= move2GPU(neunet)

if use_cuda then
	cudnn.convert(neunet, cudnn):cuda()
	cudnn.benchmark = true
	neunet:cuda()
end

print('neunet: ', neunet)

local floatSpec = ros.MsgSpec('std_msgs/Float64')

local function connect_cb(name, topic)
  print("subscriber connected: " .. name .. " (topic: '" .. topic .. "')")
end

function disconnect_cb(name, topic)
  print("subscriber diconnected: " .. name .. " (topic: '" .. topic .. "')")
end

publisher = nh:advertise("/controller", floatSpec, 100, false, connect_cb, disconnect_cb)
control   = ros.Message(floatSpec)

if not paths.dirp(opt.save)  then
  paths.mkdir(opt.save)
end

local pitch, iter = _, 1

local function log(t) 
	print('json_stats: '..json.encode(tablex.merge(t,opt,true))) 
	torch.save(opt.save .. '/' .. opt.model .. '/log.txt', json.encode(tablex.merge(t,opt,true)))
end
print('Will save at '..opt.save)

local function signum(x)
	local y
	if x < 0 
		then y = -1
	elseif x = 0
		then y = 0
	elseif x > 0
		then y = 1
	end
	return y
end

--param error is the parametric error between the reference model and the plant's output
-- y is the state variable we are trying to optimize via gradient descent/MIT rule
-- ref is the desired trajectory we want for the head 
local function compute_gains(y, ref, param_error)
	local Gamma, P, B, Lambda, K
	Kydot = -Gamma_y * y * param_error * P * B * signum(Lambda)
end

local function optimize(neunet)	
	local time = sys.clock()
	local netout, loss, grad, gradIn, u
	local w, B, net = _, _, {}    
	local params, _, _, ok = get_target()	
	if ok then 	pitch = move2GPU(torch.DoubleTensor(1):fill(params.y)) end
	local target = params.y
	neunet:zeroGradParameters()
	neunet:forget()
	iter = iter + 1

    if iter % 10  == 0 then collectgarbage() end

    pitch = pitch:cuda()
	if opt.model== 'lstm' then 
		-- print('pitch: ', pitch)
		pitch  	= {pitch}
		-- print('pitch 2: ', pitch)
		netout 	= neunet:forward(pitch)
		loss   	= cost:forward(netout[1], pitch[1])
		grad   	= cost:backward(netout[1], pitch[1])
		gradIn 	= neunet:backward(pitch, {grad})

		neunet:updateParameters(opt.lr)

		--aggregate the weights of the input and recuurent layers
		net.input_weights 	= neunet.modules[1].modules[1].gradInput
		net.prediction 		= neunet.modules[1].output
		net.recurrent_weights, net.recurrent_outputs = {}, {}
		net.iter = iter
		net.network = neunet

		for i = 1, #opt.hiddenSize do 
			table.insert(net.recurrent_weights, neunet.modules[1].recurrentModule.modules[i].gradInput)
			table.insert(net.recurrent_outputs, neunet.modules[1].recurrentModule.modules[i].outputs[1])
		end

		--W \in r^P is the weight vector of the input units
		w = net.input_weights
		B = net.recurrent_weights
		B = torch.cat(B[1], B[2], B[3])
	else
		netout = neunet:forward(pitch)
		loss 	 = cost:forward(netout, pitch)
		grad   = cost:backward(netout, pitch)
		gradIn = neunet:backward(pitch, grad)

		neunet:updateParameters(opt.lr)

		net.prediction = netout
		net.loss = loss
		net.iter = iter
		net.network = neunet
	end

	time = sys.clock() - time
	print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

	local Nf = net.prediction[1] 
	local u = -am * target + km * ref + Nf
	if opt.model=='lstm' then 
		ros.INFO("actual: %4.4f, pred: %4.8f, ref: %3.4f, loss: %4.4f, control: %4.4f", params.y, 
			        net.prediction[1], params.ref, loss, u)
	  	control.data = u
	else
		ros.INFO("actual: %d, pred: %4.4f loss: %d control: %3.4f", params.y, netout[1], loss, u)
	  	control.data = u
	end

	log{ loss = loss,
		 lr = opt.lr,
		 }

	if publisher:getNumSubscribers() == 0 then
	  print('waiting for subscriber')
	else
	  publisher:publish(control)
	end

	-- torch.save(opt.save .. '/' .. opt.model .. '/log.txt', json.encode(tablex.merge(t,opt,true)))
	saveNet(net)

	ros.spinOnce(true)
	sys.sleep(opt.rate)
end

local function main()
	local start = sys.clock()
	local finish = sys.clock() 
	print('Experiment started at: ', start)
	print('Experiment Ended at: ', finish)
	print('Total Time Taken = ', (finish - start), 'secs')

	while ros.ok() do
		optimize(neunet)
	end

	if opt.graph then 
		iterm.dot(graphgen(neunet, pitch), opt.save..'/graph.pdf')
	end
	local start = sys.clock()
	local finish = sys.clock() --os.execute('date')
	print('Experiment started at: ', start)
	print('Experiment Ended at: ', finish)
	print('Total Time Taken = ', (finish - start), 'secs')

	ros.shutdown()
end

main()



		--[[ 
		neunet.modules[1].sharedClones[1].gradInput = Input gradients
		neunet.modules[1].sharedClones[1].output = predictions
		neunet.modules[1].outputs[1] = predictions
		neunet.modules[1].output = predictions

		neunet.modules[1].gradInput = input gradients
		neunet.modules[1].modules[1].gradInput = input gradients
		neunet.modules[1].modules[1].output = prediction
		neunet.modules[1].recurrentModule.output = prediction


		print('recurrent layer 1 input weights: ', neunet.modules[1].recurrentModule.modules[1].gradInput)
		print('recurrent layer 2 input weights: ', neunet.modules[1].recurrentModule.modules[2].gradInput)
		print('recurrent layer 3 input weights: ', neunet.modules[1].recurrentModule.modules[3].gradInput)
		print('recurrent layer 1 input outs: ', neunet.modules[1].recurrentModule.modules[1].outputs[1])
		print('recurrent layer 2 input outs: ', neunet.modules[1].recurrentModule.modules[2].outputs[1])
		print('recurrent layer 3 input outs: ', neunet.modules[1].recurrentModule.modules[3].outputs[1])
		]]