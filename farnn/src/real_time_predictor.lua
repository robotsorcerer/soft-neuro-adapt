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

-- local optnet = require 'optnet'
torch.setdefaulttensortype('torch.DoubleTensor')

--options and general settings -----------------------------------------------------
opt = {
  batchSize = 1,
  data = 'data',
  gpu = 0,
  noutputs = 1,
  display = 1,
  verbose = false,
  maxIter = 20,
  silent = true,
  hiddenSize = {6, 3, 5},
  backend    = 'cunn',  --cudnn or cunn
  checkpoint = 'network/data_fastlstm-net.t7'
}

for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

if opt.display==0 then opt.display = false end

local filename = paths.concat('outputs' .. '/testlog.txt')
if paths.filep(filename) 
  then os.execute('mv ' .. filename .. ' ' .. filename ..'.old') end
testlogger = optim.Logger(paths.concat('outputs' .. '/testlog.txt'))

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

--predict outputs in real time
  local ros = require 'ros'
  ros.init('osa')
  local nh    = ros.NodeHandle()

  local spinner   = ros.AsyncSpinner()
  spinner:start()

  poseSpec    = ros.MsgSpec('ensenso/HeadPose')
  controlSpec   = ros.MsgSpec('geometry_msgs/Twist')

  pose_subscriber = nh:subscribe("/mannequine_head/pose", poseSpec, 10, { 'udp', 'tcp' }, { tcp_nodelay = true })
  u_sub       = nh:subscribe("/mannequine_head/u_valves", controlSpec, 10, { 'udp', 'tcp' }, { tcp_nodelay = true })

  poseMsg, uMsg = {}, {}
  --callbacks
  pose_subscriber:registerCallback(function(msg, header)
    -- print('\nheadPose: \n', msg)
    poseMsg.z = msg.z; poseMsg.pitch = msg.pitch; poseMsg.yaw = msg.yaw;
  end)

  u_sub:registerCallback(function(msg, header)
    -- print('\nControl law\n :', msg)
    uMsg.u1 = msg.linear.x; uMsg.u2 = msg.linear.y; uMsg.u3 = msg.linear.z; 
    uMsg.u4 = msg.angular.x; uMsg.u5 = msg.angular.y; uMsg.u6 = msg.angular.z; 
  end)

--test function-----------------------------------------------------------------
local function test(opt, model) 
  local preds;
  local iter = 0;  
  local temps = {}
  -- create mini batch            
  temps = {uMsg.u1, uMsg.u2, uMsg.u3, uMsg.u4, uMsg.u5, uMsg.u6,
            poseMsg.z, poseMsg.pitch, poseMsg.yaw} 

  inputs = torch.CudaTensor(1, 9)
  for i = 1, #temps do
    inputs[{{},{i}}] = temps[i]
  end
  -- print('inputs: ', inputs)
 
  targets = torch.CudaTensor(1,3)
  targets[{{}, {1}}] = poseMsg.z;
  targets[{{}, {2}}] = poseMsg.pitch;
  targets[{{}, {3}}] = poseMsg.yaw;
  
  -- model:forget()      --forget all past time steps
  local preds = model:forward(inputs)
  local loss  = cost:forward(preds, targets) 

  if iter % 2  == 0 then collectgarbage() end
   
  if (not opt.silent) then
    print('preds:'); print(preds); 
    print('targets: '); print(poseMsg)
    print('loss: ', loss)
    sys.sleep('0.5')
  end

  iter = iter + 1    
  -- ]]
  ros.spinOnce()
end

local function connect_cb(name, topic)
  print("subscriber connected: " .. name .. " (topic: '" .. topic .. "')")
end

local function disconnect_cb(name, topic)
  print("subscriber diconnected: " .. name .. " (topic: '" .. topic .. "')")
end

local function main() 
  local epoch = 0
  local model = init()
  while (ros.ok()) do
    --[[
        this is used to wait for subscribed 
        messages to  fully arrive else we segfault with empty inputs
      ]]
    -- print('uMsg: ', uMsg)
    epoch = epoch+1  
    -- if epoch >10 then 
      test(opt, model) 
    -- end
    ros:spinOnce()
  end
end

main()