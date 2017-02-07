--[[Source Code that implements the the learning controller described in my IEEE T-Ro Journal:
   Learning deep neural network policies for dynamic nonlinear systems 
   Olalekan Ogunmolu. IEEE International Conference on Robotics and Automation (ICRA), 2017

   Module: Glassfurnace Data: ftp://ftp.esat.kuleuven.ac.be/pub/SISTA/data/process_industry/glassfurnace.txt

   The glassfurnace data is a 3 inputs six outputs system with the following properties:

       1. Number of samples: 
        1247 samples
       2. Inputs:
        a. heating input
            b. cooling input
            c. heating input
       3. Outputs:
        a. 6 outputs from temperature sensors in a cross section of the 
        furnace

   Author: Olalekan Ogunmolu, December 2015 - May 2016
   Freely distributed under the MIT License
   ]]

-- needed dependencies
require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
-- require 'order.order_det' 
require 'sys'  
plt = require 'gnuplot' 
require 'utils.utils'
require 'utils.train'
require 'xlua'
require 'utils.model'

ros = require 'ros'

-- paths.dofile('ros/publish_net')

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
local cmd = torch.CmdLine()
cmd:text()
cmd:text('===========================================================================')
cmd:text('         Identification and Control of Nonlinear Systems Using Deep        ')
cmd:text('                      Neural Networks                                      ')
cmd:text(                                                                             )
cmd:text('             Olalekan Ogunmolu. May 2016                                 ')
cmd:text(                                                                             )
cmd:text('Code by Olalekan Ogunmolu: patlekano [at] gmail [dot] com')
cmd:text('===========================================================================')
cmd:text(                                                                             )
cmd:text(                                                                             )
cmd:text('Options')
cmd:option('-seed', 123, 'initial seed for random number generator')
cmd:option('-silent', true, 'false|true: 0 for false, 1 for true')
cmd:option('-dir', 'outputs', 'directory to log training data')

-- Model Order Determination Parameters
cmd:option('-data','data','path to -v7.3 Matlab data e.g. robotArm | glassfurnace | ballbeam | softRobot')
cmd:option('-tau', 5, 'what is the delay in the data?')
cmd:option('-m_eps', 0.01, 'stopping criterion for output order determination')
cmd:option('-l_eps', 0.05, 'stopping criterion for input order determination')
cmd:option('-trainStop', 0.5, 'stopping criterion for neural net training')
cmd:option('-sigma', 0.01, 'initialize weights with this std. dev from a normally distributed Gaussian distribution')

--Gpu settings
cmd:option('-gpu', 0, 'which gpu to use. -1 = use CPU; >=0 use gpu')
cmd:option('-backend', 'cunn', 'nn|cudnn')

-- Neural Network settings
cmd:option('-learningRate',1e-3, 'learning rate for the neural network')
cmd:option('-learningRateDecay',1e-3, 'learning rate decay to bring us to desired minimum in style')
cmd:option('-momentum', 0.9, 'momentum for sgd algorithm')
cmd:option('-model', 'lstm', 'mlp|lstm|linear|rnn')
cmd:option('-gru', false, 'use Gated Recurrent Units (nn.GRU instead of nn.Recurrent)')
cmd:option('-fastlstm', true, 'use LSTMS without peephole connections?')
cmd:option('-netdir', 'network', 'directory to save the network')
cmd:option('-optimizer', 'mse', 'mse|sgd')
cmd:option('-coefL1',   0.1, 'L1 penalty on the weights')
cmd:option('-coefL2',  0.2, 'L2 penalty on the weights')
cmd:option('-plot', false, 'true|false')
cmd:option('-maxIter', 50, 'max. number of iterations; must be a multiple of batchSize')

-- RNN/LSTM Settings 
cmd:option('-rho', 5, 'length of sequence to go back in time')
cmd:option('-dropout', true, 'apply dropout with this probability after each rnn layer. dropout <= 0 disables it.')
cmd:option('-dropoutProb', 0.35, 'probability of zeroing a neuron (dropout probability)')
cmd:option('-rnnlearningRate',1e-3, 'learning rate for the reurrent neural network')
cmd:option('-decay', 0, 'rnn learning rate decay for rnn')
cmd:option('-batchNorm', false, 'apply szegedy and Ioffe\'s batch norm?')
cmd:option('-ninputs', 9, 'size of input layer being fed from system')
cmd:option('-noutputs', 3, 'size of linear output layer after rnn')
cmd:option('-batchSize', 100, 'Batch Size for mini-batch training')

-- Print options
cmd:option('-print', false, 'false = 0 | true = 1 : Option to make code print neural net parameters')  -- print System order/Lipschitz parameters
cmd:option('-weights', false, 'false = 0 | true = 1 : Option to make code print neural net weights')
--vicon settings
cmd:option('-ros', false, 'initialize ros engine and publish neural network?')
cmd:option('-vicon', true, 'is vicon on?')
cmd:option('-publishTransform', false, 'publish transform?')
cmd:option('-publishTwist', true, 'publish twist?')

-- misc
opt = cmd:parse(arg or {})
torch.manualSeed(opt.seed)
torch.setnumthreads(8)

if not opt.silent then
   print(opt)
end

opt.hiddenSize = {opt.ninputs, 5, 3}

if (opt.model == 'rnn') then  
  rundir = cmd:string('rnn', opt, {dir=true})
elseif (opt.model == 'mlp') then
  rundir = cmd:string('mlp', opt, {dir=true})
elseif (opt.model == 'lstm')  then 
  if (opt.fastlstm) then
    rundir = cmd:string('fastlstm', opt, {dir=true})  
  elseif (opt.gru) then
    rundir = cmd:string('gru', opt, {dir=true})
  else
    rundir = cmd:string('lstm', opt, {dir=true})
  end
else
  assert("you have entered an invalid model") 
end

print('rundir', rundir)

--log to file
opt.rundir = opt.dir .. '/' .. rundir
print('opt.rundir', opt.rundir)
if paths.dirp(opt.rundir) then
  os.execute('rm -r ' .. opt.rundir)
end
os.execute('mkdir -p ' .. opt.rundir)
-- cmd:addTime(tostring(opt.rundir) .. ' Deep Head Motion Control', '%F %T')
cmd:text()
cmd:log(opt.rundir .. '/log.txt', opt)

logger = optim.Logger(paths.concat(opt.rundir .. '/log.txt'))
testlogger = optim.Logger(paths.concat(opt.rundir .. '/testlog.txt'))
-------------------------------------------------------------------------------
-- Fundamental initializations
-------------------------------------------------------------------------------
print("")
print('==> fundamental initializations')
if opt.gpu == -1 then torch.setdefaulttensortype('torch.FloatTensor') end

data        = 'data/' .. opt.data
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

print(sys.COLORS.red .. '==> Parsing raw data')
splitData = {}
splitData = split_data(opt)

print(sys.COLORS.red .. '==> Data Pre-processing')
kk          = splitData.train_input[1]:size(1)
--==================================================================================================
--[[@ToDo: Determine input-output order using He and Asada's prerogative]]
print(sys.COLORS.red .. '==> Determining input-output model order parameters' )

--[[
find optimal # of input variables from data
qn  = computeqn(train_  input, train_out[3])

compute actual system order
utils = require 'order.utils'
inorder, outorder, q =  computeq(train_input, (train_out[3])/10, opt)
]]

--------------------utils--------------------------------------------------------------------------
print(sys.COLORS.red .. '==> Constructing neural network')
---------------------------------------------------------------------------------------------------
transfer    =  nn.ReLU()  

paths.dofile('utils/model.lua')
cost, neunet          = contruct_net()

print(neunet)
print('\t\n\t|\n\t|\n\tv \n\tNetwork Table\n'); 

-- retrieve parameters and gradients
parameters, gradParameters = neunet:getParameters()
print(string.format('net params: %d, gradParams: %d', parameters:size(1), gradParameters:size(1)))
--=====================================================================================================
neunet = transfer_data(neunet)  
cost = transfer_data(cost)
local function weights_init(m)
   local name = torch.type(m)
   if name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
    else
      m.weight:normal(0.0, 0.02)
      m:noBias()
   end
end

neunet:apply(weights_init)

print '==> configuring optimizer\n'

 if opt.optimizer == 'mse' then
   optimMethod = msetrain
   
   -- Perform SGD step:
 elseif opt.optimizer == 'sgd' then   
     sgdState = {
        learningRate = 1e-2,
        learningRateDecay = 1e-6,
        momentum = 0,
        weightDecay = 0
      }
   optimMethod = optim.sgd

 else  
   error(string.format('Unrecognized optimizer "%s"', opt.optimizer))  
 end

local function connect_cb(name, topic)
  print("subscriber connected: " .. name .. " (topic: '" .. topic .. "')")
end

local function disconnect_cb(name, topic)
  print("subscriber diconnected: " .. name .. " (topic: '" .. topic .. "')")
end

if opt.ros then 
  --init ros engine---------------------------------------------------------------
  print('==> ros publisher initializations')
  ros.init('soft_robot')
  spinner = ros.AsyncSpinner()
  spinner:start()

  local nh = ros.NodeHandle()
  local neural_weights = ros.MsgSpec('std_msgs/String')

  if(opt.vicon) then   
    print(sys.COLORS.red .. '==> initiating vicon publishers and subscribers')
    -- paths.dofile('ros/publish_net.lua')
    -- paths.dofile('ros/vicon_sub.lua') 
    transform_subscriber = nh:subscribe("/vicon/Superdude/root", 'geometry_msgs/TransformStamped', 10, { 'udp', 'tcp' }, { tcp_nodelay = true })
    twist_subscriber     = nh:subscribe("/vicon/headtwist",      'geometry_msgs/Twist',            10, { 'udp', 'tcp' }, { tcp_nodelay = true })

    if opt.publishTransform then  --note that the super_move package publishes the head twist by default
      -- register a callback function that will be triggered from ros.spinOnce() when a message is available.
      transform_subscriber:registerCallback(function(msg, header)
        print('Header:')
        print(header)
        print('Message:')
        print(msg)
      end)
    end

    if opt.publishTwist then
      twist_subscriber:registerCallback(function(msg, header)
        if not opt.silent then print('Head twist: \n', msg) end
        end)
    end
  end

  pub = nh:advertise("neural_net", neural_weights, 100, false, connect_cb, disconnect_cb)
  ros.spinOnce()

  msg = ros.Message(neural_weights)
end

-------------------------------------------------------------------------------

print '==> defining training procedure'
-------------------------------------------------------------------------------
logger:setNames{'loss'}
local function train(data)    
  --time we started training
  local time = sys.clock()
  --track the epochs
  epoch = epoch or 1
  iter = iter or 0

  if opt.model == 'rnn' then 
    train_rnn(opt) 
    -- time taken for one epoch
    time = sys.clock() - time
    time = time / height
    -- print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

    if epoch % 2 == 0 then
      saveNet()
    end

    if opt.ros then 
      if pub:getNumSubscribers() == 0 then
        print('please subscribe to the /neural_net topic')
      else
        msg.data = neunet
        pub:publish(msg)
      end

      sys.sleep(0.01)
      ros.spinOnce()
    end

    -- next epoch
    epoch = epoch + 1

  elseif (opt.model == 'lstm') then
    train_lstm(opt)
    -- time taken for one epoch
    time = sys.clock() - time
    time = time / height
    -- print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')
    
    if epoch % 2 == 0 then
      saveNet(epoch)
    end

    if opt.ros then 
      if pub:getNumSubscribers() == 0 then
        print('please subscribe to the /neural_net topic')
      else
        print('publishing neunet weights: ', neunet)
        local weights, biases
        weights = neunet.modules[1].recurrentModule.modules[7].weight
        biases  = neunet.modules[1].recurrentModule.modules[7].bias
        -- print(weights, 'weights')
        msg.data = tostring(weights)
        pub:publish(msg)
      end

      sys.sleep(0.01)
      ros.spinOnce()
    end

    -- next epoch
    epoch = epoch + 1

  elseif  opt.model == 'mlp'  then
    train_mlp(opt)
    -- time taken for one epoch
    time = sys.clock() - time
    time = time / height
    print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')
    
    if epoch % 10 == 0 then      
      saveNet()
    end

    if opt.ros then
      if pub:getNumSubscribers() == 0 then
        print('please subscribe to the /neural_net topic')
      else
        msg.data = neunet
        pub:publish(msg)
      end
      
      sys.sleep(0.01)
      ros.spinOnce()
    end

    -- next epoch
    epoch = epoch + 1
  else
    print("Incorrect model entered")
  end            
end     


function saveNet(epoch)
  --check if network directory exists
  if not paths.dirp('network')  then
    paths.mkdir('network')
  end
  -- save/log current net
  if opt.model == 'rnn' then
    netname = opt.data .. '_' .. epoch .. '_rnn-net.t7'
  elseif opt.model == 'mlp' then
    netname = opt.data .. '_' .. epoch .. '_mlp-net.t7'
  elseif opt.model == 'lstm' then
    if opt.gru then
      netname = opt.data .. '_gru-net.t7'
    elseif opt.fastlstm then
      netname = opt.data .. --[['_'.. epoch ..]] '_fastlstm-net.t7'
    else
      netname = opt.data .. '_' .. epoch .. '_lstm-net.t7'
    end
  else
    netname = opt.data .. '_' .. epoch .. '_neunet.t7'
  end  
  
  local filename = paths.concat(opt.netdir, netname)
  if epoch == 0 then
    if paths.filep(filename) then
     os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
    end  
    os.execute('mkdir -p ' .. sys.dirname(filename))
  else    
    -- print('<trainer> saving network model to '..filename)
    torch.save(filename, neunet)
  end
end

local function main()

  local start = sys.clock() --os.execute('date');

  -- if (ros.ok()) then
    for i = 1, 50 do
      train(trainData)
    end
  -- end

  ---now test the trained model on testing data
  paths.dofile('test.lua')

  local finish = sys.clock() --os.execute('date')
  print('Experiment started at: ', start)
  print('Experiment Ended at: ', finish)
  print('Total Time Taken = ', (finish - start), 'secs')

  -- transform_subscriber:shutdown()
  -- twist_subscriber:shutdown()
  -- ros.shutdown()
end

main()

