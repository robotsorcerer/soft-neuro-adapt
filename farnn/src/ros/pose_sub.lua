--[[
	This code subscribes to the twist motion of the head from vicon_listener that I wrote. 
	It also subscribes to the geometry/TransformStamped message from vicon bridge
	The twist motion is what we pass to the neyral network to develo a real-time function approximator

	Author: Olalekan Ogunmolu.
	Date:   October 2016
	Lab Affiliation: Gans' Lab UT Dallas
	]]

local opt = {
	publishTransfrom = false,
	publishTwist = false
}

local ros 	= require 'ros'

ros.init('ensenso_sub')

spinner 	= ros.AsyncSpinner()
spinner:start()

nodehandle 	= ros.NodeHandle()
poseSpec 	= ros.MsgSpec('ensenso/HeadPose')
controlSpec = ros.MsgSpec('geometry_msgs/Twist')


local pose_subscriber = nodehandle:subscribe("/mannequine_head/pose", poseSpec, 10, { 'udp', 'tcp' }, { tcp_nodelay = true })
local u_sub  		  = nodehandle:subscribe("/mannequine_head/u_valves",controlSpec, 10, { 'udp', 'tcp' }, { tcp_nodelay = true })

-- subscribe to vicon_receiver topic with 10 messages back-log
-- transport_options (arguments 4 & 5) are optional - used here only for demonstration purposes
poseMsg, uMsg, inputs = {}, {}, {}
-- register a callback function that will be triggered from ros.spinOnce() when a message is available.
pose_subscriber:registerCallback(function(msg, header)
  -- print('\nheadPose: \n', msg)
  	poseMsg.z = msg.z; poseMsg.pitch = msg.pitch; poseMsg.yaw = msg.yaw;
end)

u_sub:registerCallback(function(msg, header)
  uMsg.u1 = msg.linear.x; uMsg.u2 = msg.linear.y; uMsg.u3 = msg.linear.z; 
  uMsg.u4 = msg.angular.x; uMsg.u5 = msg.angular.y; uMsg.u6 = msg.angular.z; 
end)

if opt.publishTwist then
	twist_subscriber:registerCallback(function(msg, header)
		if not opt.silent then print('Head twist: \n', msg) end
		end)
end

-- function table_merge(t1, t2)
--    for k,v in ipairs(t1) do
--       table.insert(inputs, v)
--    end 
--    for k,v in ipairs(t1) do
--       table.insert(inputs, v)
--    end 
 
--    return t1
-- end

while ros.ok() do
  inputs = {uMsg.u1, uMsg.u2, uMsg.u3, uMsg.u4, uMsg.u5, uMsg.u6,
			poseMsg.z, poseMsg.pitch, poseMsg.yaw}
  print(inputs)
  ros.spinOnce()
  sys.sleep(0.1)
end

pose_subscriber:shutdown()

ros.shutdown()