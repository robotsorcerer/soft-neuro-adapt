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

local ros = require 'ros'

ros.init('ensenso_sub')

spinner = ros.AsyncSpinner()
spinner:start()

nodehandle = ros.NodeHandle()
string_spec = ros.MsgSpec('ensenso/HeadPose')

local pose_subscriber = nodehandle:subscribe("/mannequine_head/pose", 'ensenso/HeadPose', 10, { 'udp', 'tcp' }, { tcp_nodelay = true })

print(pose_subscriber)
-- subscribe to vicon_receiver topic with 10 messages back-log
-- transport_options (arguments 4 & 5) are optional - used here only for demonstration purposes

-- register a callback function that will be triggered from ros.spinOnce() when a message is available.
pose_subscriber:registerCallback(function(msg, header)
  print('Header:')
  print(header)
  print('Message:')
  print(msg)
end)

if opt.publishTwist then
	twist_subscriber:registerCallback(function(msg, header)
		if not opt.silent then print('Head twist: \n', msg) end
		end)
end

while ros.ok() do
  ros.spinOnce()
  sys.sleep(0.1)
end

pose_subscriber:shutdown()

ros.shutdown()