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
  batchSize     = 1,
  data          = 'data',
  gpu           = 0,
  noutputs      = 3,
  display       = 1,
  verbose       = true,
  maxIter       = 20,
  silent        = true,
  useVicon      = true,
  squash        = true,
  model         = 'lstm',
  real_time_net = true, --use real-time network approximator
  hiddenSize    = {9, 6, 6},
  seed          = 123,
  backend       = 'cunn',  --cudnn or cunn
  checkpoint    = 'network/data_fastlstm-net.t7'
}

for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

if opt.display==0 then opt.display = false end

local filename = paths.concat('outputs' .. '/testlog.txt')
if paths.filep(filename) 
  then os.execute('mv ' .. filename .. ' ' .. filename ..'.old') end
testlogger = optim.Logger(paths.concat('outputs' .. '/testlog.txt'))

noutputs  = 6 --opt.noutputs
torch.manualSeed(opt.seed)

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
  local model, cost
  if opt.real_time_net then
    paths.dofile('utils/model.lua')
    cost, model = contruct_net()
  else    
    model   = torch.load(opt.checkpoint)
    cost    = nn.SequencerCriterion(nn.MSECriterion()):cuda()
  end

  assert(model ~='', 'must provide a trained model')

  if use_cuda then
    model:cuda()
    model:evaluate()
  else
    model:double()
    model:evaluate()
  end

  netmods = model.modules;
  --------------------------------------------------------------------------------
  return cost, model
end

local function transfer_data(x)
  if use_cuda then
    x:cuda()
  else 
    x:float()
  end
  return x
end

-- testlogger:setNames{'iter', 'loss'}

--[[
  Fundamental ROS Initializations
  predict outputs in real time
]]
print (sys.COLORS.red .. '==> Fundamental ROS Initializations')
local ros = require 'ros'
ros.init('osa')

spinner   = ros.AsyncSpinner()
spinner:start()

nh          = ros.NodeHandle()

poseSpec    = {}

if opt.useVicon then 
  poseSpec  = ros.MsgSpec('geometry_msgs/Pose')
else
  poseSpec    = ros.MsgSpec('ensenso/HeadPose')
end

controlSpec = ros.MsgSpec('geometry_msgs/Twist')
pred_spec   = ros.MsgSpec('geometry_msgs/Pose')
loss_spec   = ros.MsgSpec('std_msgs/Float64')
net_spec    = ros.MsgSpec('std_msgs/Float64MultiArray')

--subscribers from nn_controller.cpp
local pose_subscriber = nh:subscribe("/mannequine_head/pose", poseSpec, 10, { 'udp', 'tcp' }, { tcp_nodelay = true })
local u_sub       = nh:subscribe("/mannequine_head/u_valves", controlSpec, 10, { 'udp', 'tcp' }, { tcp_nodelay = true })
--advertisers to nn_controller.cpp
local pred_pub    = nh:advertise("/mannequine_pred/preds",    pred_spec, 100, false, connect_cb, disconnect_cb)
local net_pub     = nh:advertise("mannequine_pred/net_weights", net_spec, 10)
local bias_pub     = nh:advertise("mannequine_pred/net_biases", net_spec, 10)
local loss_pub    = nh:advertise('/mannequine_pred/net_loss', loss_spec, 10)
--messages for subscribe
poseMsg, uMsg, inputs  = {}, {}, {}
--msgs for publishers
pred_msg = ros.Message(pred_spec)
loss_msg = ros.Message(loss_spec)

--callbacks
pose_subscriber:registerCallback(function(msg, header)
  -- print('\nheadPose: \n', msg)
  if(not opt.useVicon) then
    poseMsg.z = msg.z; poseMsg.pitch = msg.pitch; poseMsg.roll = msg.roll; 
                      poseMsg.yaw = msg.yaw;
  else
    poseMsg.z = msg.position.z; 
    poseMsg.roll = msg.orientation.x;
    poseMsg.pitch = msg.orientation.y;
    poseMsg.yaw   = msg.orientation.z;
  end
end)

u_sub:registerCallback(function(msg, header)
  -- print('\nControl law\n :', msg)
  uMsg.u1 = msg.linear.x; uMsg.u2 = msg.linear.y; uMsg.u3 = msg.linear.z; 
  uMsg.u4 = msg.angular.x; uMsg.u5 = msg.angular.y; uMsg.u6 = msg.angular.z; 
end)

local function connect_cb(name, topic)
  print("subscriber connected: " .. name .. " (topic: '" .. topic .. "')")
end

local function disconnect_cb(name, topic)
  print("subscriber diconnected: " .. name .. " (topic: '" .. topic .. "')")
end

function tensorToMsg(tensor)
  local msg = ros.Message(net_spec)
  msg.data = tensor:reshape(tensor:nElement())
  for i=1,tensor:dim() do
    local dim_desc = ros.Message('std_msgs/MultiArrayDimension')
    dim_desc.size = tensor:size(i)
    dim_desc.stride = tensor:stride(i)
    table.insert(msg.layout.dim, dim_desc)
  end
  return msg
end

--this gives the recurrent modules within the saved net
local function netToMsg(model)
  --network weights
  -- netmods     = model.modules;
  -- weights,biases  = {}, {};
  netparams   = {}
  local broadcast_weights, broadcast_biases = {}, {}
  local netmods     = model.modules
  local modules   = netmods[1].recurrentModule.modules
  --[[e.g. modules[1] will be nn.FastLSTM(9 -> 9)
      modules[2] will be nn.Dropout(0.3,busy)
      etc
    ]]
  local length  = #modules
  for i = 1, length do
    netparams[i]  = {['weight']=modules[i].weight, ['bias']=modules[i].bias}
  end

  -- find indices in netparams that contain weights
  for k, v in pairs(netparams) do 
    if netparams[k].weight then 
       broadcast_weights = netparams[k].weight
       broadcast_biases  = netparams[k].bias
    end
  end

  br_weights = broadcast_weights:double()
  br_biases  = broadcast_biases:double()

  -- print('br_weights: \n', br_weights)
  -- print('br_biases: \n', br_biases)
  --[[concatenate the weights and biases before publishing
  --For recurrent modules,note that the first three columns 
  will be the weights while the last one 
  column will be the bias]]
  local netmsg = torch.cat({br_weights, br_biases}, 2)

  wgts_params = tensorToMsg(br_weights)
  bias_params = tensorToMsg(br_biases)
  return wgts_params, bias_params
end

--[[we need to strip the torch.typeTensor before
publishing our pred msgs]]
local dump = function(vec)
    vec = vec:view(vec:nElement())
    local t = {}
    for i=1,vec:nElement() do
        x = string.format('%.8f', vec[i])
        x = tonumber(x)
        table.insert(t, x)
    end
    return t
end

--[[Convenience printing function from tensor to table]]
local tensorToNice = function(vec)
    vec = vec:view(vec:nElement())
    local t = {}
    for i=1,vec:nElement() do
        t[#t+1] = string.format('%.8f', vec[i])
    end
    return table.concat(t, '  ')
end

local geometryPoseToNice = function(poseTab)
  local t = {}
  x1 = poseTab.position.x
  y1 = poseTab.position.y
  z1 = poseTab.position.z

  x2 = poseTab.orientation.x
  y2 = poseTab.orientation.y
  z2 = poseTab.orientation.z
  return table.concat(t, x1, '  ', y1, '  ', z1, '  ', x2, '  ', y2, '  ', z2)
end

--construct sigmoiud function to squash network outputs to range of valves
sigmoid = nn.Sequential()
sigmoid:add(nn.Linear(1, 1))
sigmoid:add(nn.Sigmoid())

local function squash_sigmoid(x)
  local xx = torch.DoubleTensor(1):fill(x)
  local res = sigmoid:forward(xx)
  -- print('xx: ', xx, 'res: ', res)
  return res
end

local function squash_pvq(x)
  local xx = torch.DoubleTensor(1):fill(x)
  local res = 400*sigmoid:forward(xx)
  return res
end


local function main() 
  local epoch = 0
  local cost, model = init()
  local params, gradParams = model:getParameters()
  print('params', params:size())
  print('gradParams', gradParams:size())
  --initialize weight matrices from a normal randon distribution
  params:copy(torch.randn(params:size()))
  -- sys.sleep(10)
  local pred_table = {}

  if(opt.real_time_net) then
    transfer_data(cost)
    transfer_data(model)
  end

  while (ros.ok()) do

    epoch = epoch+1  
    local iter = 0;  

    if epoch > 10 then  --test if messages have arrived
      local preds;

      local inputs  = torch.CudaTensor(1, 9)
      local targets = torch.CudaTensor(1,6)

      print("poseMsg: ", poseMsg)
      print("uMsg: ", uMsg)

      inputs[{{},{1}}] = uMsg.u1
      inputs[{{},{2}}] = uMsg.u2
      inputs[{{},{3}}] = uMsg.u3
      inputs[{{},{4}}] = uMsg.u4
      inputs[{{},{5}}] = uMsg.u5
      inputs[{{},{6}}] = uMsg.u6
      inputs[{{},{7}}] = poseMsg.z
      inputs[{{},{8}}] = poseMsg.pitch
      inputs[{{},{9}}] = poseMsg.roll

      targets[{{}, {1}}] = poseMsg.pitch 
      targets[{{}, {2}}] = poseMsg.pitch
      targets[{{}, {3}}] = poseMsg.roll
      --augment the nontracted states with 0
      targets[{{}, {4}}] = poseMsg.roll
      targets[{{}, {5}}] = poseMsg.z

      targets[{{}, {6}}] = poseMsg.z

      -- print('inputs: ', inputs)  
      -- print('targets: ', targets)  

      if opt.real_time_net then
        model:zeroGradParameters()
      end
        model:forget()  --forget all past time steps

      local preds         = model:forward(inputs)
      local loss          = cost:forward(preds, targets) 
      local gradOutputs   = cost:backward(preds, targets)
      local gradInputs    = neunet:backward(inputs, gradOutputs) 

      model:updateParameters(5e-3)

      if iter % 2  == 0 then collectgarbage() end
      

      testlogger:add{loss}

      --populate msgs for ros publishers
      --serialize preds msg

      pred_table = dump(preds)
      --squash dakota valve inputs to 0 and 1 duty cycle
      if(squash) then
        pred_table[1] = dump(squash_sigmoid(pred_table[1]))[1]
        pred_table[3] = dump(squash_sigmoid(pred_table[3]))[1]
        pred_table[5] = dump(squash_sigmoid(pred_table[5]))[1]
        pred_table[6] = dump(squash_sigmoid(pred_table[6]))[1]
        --squash pvq valves
        pred_table[2] = dump(squash_pvq(pred_table[2]))[1]  --pvq1
        pred_table[4] = dump(squash_pvq(pred_table[4]))[1]  --pvq1
      end

      pred_msg.position.x = pred_table[1]
      pred_msg.position.y = pred_table[3]
      pred_msg.position.z = pred_table[5]
      pred_msg.orientation.x = pred_table[6]
      pred_msg.orientation.y = pred_table[2]
      pred_msg.orientation.z = pred_table[4]

      if opt.verbose then
        print('preds: ', pred_table)--geometryPoseToNice(pred_msg)); 
        print('targets: ', tensorToNice(targets))
        print('loss: ', loss)
      end

      --serialize loss msgs
      loss_msg.data = loss
      --publish net weights
      local wgts_params, bias_params  = netToMsg(model)

      --publish all the messages
      pred_pub:publish(pred_msg)
      loss_pub:publish(loss_msg)
      net_pub:publish(wgts_params)
      bias_pub:publish(bias_params)

      iter = iter + 1     
    end
    ros.spinOnce()
    sys.sleep(0.1)
  end
end

main()