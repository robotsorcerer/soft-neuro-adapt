--[[ This file measures how generalizable the network model is.
      Author: Olalekan Ogunmolu. 
      Date:   Oct 20, 2016
      Lab Affiliation: Gans' Lab
      ]]
require 'torch'
require 'data.dataparser'
require 'rnn'
require 'nngraph'
nn.LSTM.usenngraph = true -- faster
require 'optim'

local optnet = require 'optnet'
torch.setdefaulttensortype('torch.FloatTensor')

--options and general settings -----------------------------------------------------
opt = {
  batchSize = 5,
  data = 'data',
  gpu = 0,
  ros = true,
  noutputs = 1,
  display = 1,
  verbose = false,
  maxIter = 20,
  silent = false,
  hiddenSize = {6, 3, 5},
  backend    = 'cunn',  --cudnn or cunn
  checkpoint = 'network/data_4_fastlstm-net.t7'
}

for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

if opt.display==0 then opt.display = false end

local filename = paths.concat('outputs' .. '/testlog.txt')
if paths.filep(filename) 
  then os.execute('mv ' .. filename .. ' ' .. filename ..'.old') end
testlogger = optim.Logger(paths.concat('outputs' .. '/testlog.txt'))

local spinner,nh

local function init()

  --GPU Settings -----------------------------------------------------------------------
  use_cuda = false

  local msg
  if opt.gpu >= 0 then
    require 'cutorch'
    require 'cunn'
    if opt.backend == 'cudnn' then require 'cudnn' end
    use_cuda = true
    cutorch.setDevice(opt.gpu + 1)
    msg = string.format('Code deployed on GPU %d with cudnn backend', opt.gpu+1)
  else
    msg = 'Running in CPU mode'
    require 'nn'
  end
  if opt.verbose == 1 then print(msg) end
  -------------------------------------------------------------------------------

  ---Net Loader---------------------------------------------------------------------------
  local model
  model = torch.load(opt.checkpoint)

  require 'rnn'
  cost = nn.SequencerCriterion(nn.MSECriterion()):cuda()

  assert(model ~='', 'must provide a trained model')

  if use_cuda then
    model:cuda()
    model:evaluate()
  else
    model:double()
    model:evaluate()
  end

  netmods = model.modules;
  if opt.verbose then print('netmods: \n', netmods) end
  --------------------------------------------------------------------------------
  return model
end

local function transfer_data(x)
  if use_cuda then
    x:cuda()
  else 
    x:float()
  end
  return x
end

testlogger:setNames{'iter', 'loss'}

local pose_info = {}

if(opt.ros) then
  ros = require 'ros'
  --init ros engine---------------------------------------------------------------
  print('==> ros publisher initializations')
  ros.init('soft_robot')
  spinner = ros.AsyncSpinner()
  spinner:start()

  nh = ros.NodeHandle()

  string_spec = ros.MsgSpec('ensenso/HeadPose')
  pose_subscriber = nh:subscribe("/mannequine_head/pose", 'ensenso/HeadPose', 10, { 'udp', 'tcp' }, { tcp_nodelay = true })
    print(pose_subscriber)

  -- register a callback function that will be triggered from ros.spinOnce() when a message is available.
  pose_subscriber:registerCallback(function(msg, header)
    pose_info = msg
    print(string.format('x: %.4f, y: %.4f, z: %.4f, pch: %.4f, yaw: %.4f', 
          msg.x, msg.y, msg.z, msg.pitch, msg.yaw))
  end)
end


--test function-----------------------------------------------------------------
local function predict(opt, model)
  -- subscribe to vicon_receiver topic with 10 messages back-log
  -- transport_options (arguments 4 & 5) are optional - used here only for demonstration purposes

 -- local vars
 local splitData = {}; 
 splitData = split_data(opt)
 local time = sys.clock()
 local testHeight = splitData.test_input[1]:size(1)

  local preds;
  local iter = 0;  

  -- create mini batch        
  local inputs, targets = {}, {}      
  --[[ _, _, inputs, targets = get_datapair(opt, t)
  inputs = torch.cat({inputs[1], inputs[2], inputs[3], inputs[4], inputs[5],
                      inputs[6], inputs[7], inputs[8], inputs[9]}, 2)  
  targets = torch.cat({targets[1], targets[2], targets[3]}, 2) 
  ]]
  inputs = {pose_info.x, pose_info.y, pose_info.z, 
              pose_info.pitch, pose_info.yaw}

  -- test samples
  model:forget()  --forget all past time steps
  local preds = model:forward(inputs)
  local loss = cost:forward(preds, targets) 

  local  gradOutputs  = cost:backward(preds, targets)
  -- local gradInputs    = model:backward(inputs, gradOutputs) 
  model:updateParameters(5e-3)

  if iter % 2  == 0 then collectgarbage() end
   
  -- timing
  time = sys.clock() - time
  time = time / testHeight

  testlogger:add{t, loss}
  if (not opt.silent) then
    print('loss'); print(loss)
    print('preds:'); print(preds); print('targets: '); print(targets)
    sys.sleep('1')
  end

  iter = iter + 1
end

local function connect_cb(name, topic)
  print("subscriber connected: " .. name .. " (topic: '" .. topic .. "')")
end

local function disconnect_cb(name, topic)
  print("subscriber diconnected: " .. name .. " (topic: '" .. topic .. "')")
end

local function main()  
  local model = init()
    while(ros.ok()) do
      predict(opt, model)
      ros.spinOnce()
      sys.sleep(0.01)
    end

    pose_subscriber:shutdown()

    ros.shutdown()
end

main()