require 'torch'
require 'nn'
require 'rnn'
require 'nngraph'

local cmd = torch.CmdLine()
local ros = require 'ros'

--Gpu settings
cmd:option('-gpu', 0, 'which gpu to use. -1 = use CPU; >=0 use gpu')
cmd:option('-checkpoint', 'sr-net/softRobot_lstm-net.t7', 'load the trained network e.g. <lstm-net.t7| rnn-net.t7|mlp-net.t7>')
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-ros', true, 'use ros?')
cmd:option('-floatarray', false, 'use float multiarray instead of string?')
cmd:option('-verbose', false)

local opt = cmd:parse(arg)
torch.setnumthreads(8)

local use_cuda, msg = false
if opt.gpu >= 0  and opt.backend == 'cudnn' then
	require 'cunn'
	require 'cutorch'
	require 'cudnn'
	use_cuda = true
	cutorch.setDevice(opt.gpu + 1)
	out = string.format('Code deployed on GPU %d with cudnn backend', opt.gpu+1)
else
  out = 'Running in CPU mode'
  require 'nn'
end
if opt.verbose then print(out) end

local checkpoint, model

if use_cuda then
	model = torch.load(opt.checkpoint)
	model:cuda()
else
	model = torch.load(opt.checkpoint)
	model:double()
end

model:evaluate()

netmods = model.modules;

weights,biases = {}, {};
netparams= {}
local broadcast_weights = {}

ros.init('advertise_neunet')
local nh = ros.NodeHandle()

local spinner = ros.AsyncSpinner()
spinner:start()

local string_spec, publisher, subscriber

if opt.floatarray then 
	publisher = nh:advertise("multi_net", 'std_msgs/Float64MultiArray', 10)
	string_spec = ros.MsgSpec('std_msgs/Float64MultiArray')
else
	string_spec = ros.MsgSpec('std_msgs/String')
	publisher = nh:advertise("saved_net", string_spec, 10, false, connect_cb, disconnect_cb)
end
ros.spinOnce()

function tensorToMsg(tensor)
  local msg = ros.Message(string_spec)
  msg.data = tensor:reshape(tensor:nElement())
  for i=1,tensor:dim() do
    local dim_desc = ros.Message('std_msgs/MultiArrayDimension')
    dim_desc.size = tensor:size(i)
    dim_desc.stride = tensor:stride(i)
    table.insert(msg.layout.dim, dim_desc)
  end
  return msg
end


if #netmods == 1 then   		--recurrent modules
	local modules 	= netmods[1].recurrentModule.modules
	local length 	= #modules
	for i = 1, length do
		netparams[i] 	= {['weight']=modules[i].weight, ['bias']=modules[i].bias}
	end

	-- find indices in netparams that contain weights
	for k, v in pairs(netparams) do 
		if netparams[k].weight then 
		   broadcast_weights = netparams[k].weight
		end
	end

elseif #netmods > 1 then   --mlp modules
	for i = 1, #netmods do		
		netparams[i] 	= {['weight']=netmods[i].weight, ['bias']=netmods[i].bias}
	end

	-- find indices in netparams that contain weights
	for k, v in pairs(netparams) do 
		if netparams[k].weight then 
		   broadcast_weights = netparams[k].weight
		end
	end
end

if opt.verbose then 
	print('\nbroadcast_weights\n:'); 
	print(broadcast_weights)
end

br_weights = broadcast_weights:double()

while(ros.ok())	do
 	if publisher:getNumSubscribers() == 0  then
    	print('waiting for subscriber')
  	else
    	if opt.floatarray then 
    		--publish multiarray
    		params = tensorToMsg(br_weights)
    		publisher:publish(params)
    	else    		 --publish string
    		params 	  = ros.Message(string_spec)
    		params.data = tostring(br_weights)   
    		publisher:publish(params)
    	end
  	end

  	ros.spinOnce()
  	sys.sleep(0.1)
end

ros.shutdown()

