--[[
This publishes the network as a ros topic;
The topic can be retrieved in labview using ROS for LabVIEW

Author: Olalekan Ogunmolu
		October 2016
]]
--[[
ros = require 'ros'

--init ros engine---------------------------------------------------------------
print('==> ros publisher initializations')
ros.init('soft_robot')
local spinner = ros.AsyncSpinner()
spinner:start()

local nh = ros.NodeHandle()
local neural_weights = ros.MsgSpec('std_msgs/String')

local pub = nh:advertise("neural_net", neural_weights, 100, false)
ros.spinOnce()

msg = ros.Message(neural_weights)

function tensorToMsg(tensor)
  local msg = ros.Message(specFloat64MultiArray)
  msg.data = tensor:reshape(tensor:nElement())
  for i=1,tensor:dim() do
    local dim_desc = ros.Message('std_msgs/MultiArrayDimension')
    dim_desc.size = tensor:size(i)
    dim_desc.stride = tensor:stride(i)
    table.insert(msg.layout.dim, dim_desc)
  end
  return msg
end

]]

--init ros engine---------------------------------------------------------------

nh = ros.NodeHandle()

-- recWpub = nh:advertise("recWeights", neural_weights, 100, false)
-- recBpub = nh:advertise("recBiases", neural_weights, 100, false)
outWpub = nh:advertise("/output_weights", 'std_msgs/Float64MultiArray', 5, false)
-- outBpub = nh:advertise("outBiases", neural_weights, 100, false)

neural_weights = ros.MsgSpec('std_msgs/Float64MultiArray')

ros.spinOnce()

-- rwmsg = ros.Message(neural_weights)
-- rbmsg = ros.Message(neural_weights)
-- owmsg = ros.Message(neural_weights)
-- obmsg = ros.Message(neural_weights)

function tensorToMsg(tensor)
  local msg = ros.Message(neural_weights)
  -- assert(type(tensor=='userdata'), 'message to be published must be a tensor') 
  msg.data = tensor:reshape(tensor:nElement()):float()
  for i=1,tensor:dim() do
    local dim_desc = ros.Message('std_msgs/MultiArrayDimension')
    -- if tensor:dim() == 1 then
    --  dim_desc[i].label = "height" 
    -- elseif tensor:dim() == 2 then
    --  dim_desc[0].label = "height"  
    --  dim_desc[1].label = "width"
    -- elseif tensor:dim() == 3 then
    --  dim_desc[0].label = "height"  
    --  dim_desc[1].label = "width"
    --  dim_desc[2].label = "channel" 
    -- end

    dim_desc.size = tensor:size(i)
    dim_desc.stride = tensor:stride(i)
    table.insert(msg.layout.dim, dim_desc)
  end
  return msg
end