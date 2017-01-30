--[[
	This code subscribes to the twist motion of the head from vicon_listener that I wrote. 
	It also subscribes to the geometry/TransformStamped message from vicon bridge
	The twist motion is what we pass to the neyral network to develo a real-time function approximator

	Author: Olalekan Ogunmolu.
	Date:   October 2016
	Lab Affiliation: Gans' Lab UT Dallas
	]]


local ros = require 'ros'

-- ros.init('vicon_sub')

-- spinner = ros.AsyncSpinner()
-- spinner:start()

nodehandle = ros.NodeHandle()

local transform_subscriber = nodehandle:subscribe("/vicon/Superdude/root", 'geometry_msgs/TransformStamped', 10, { 'udp', 'tcp' }, { tcp_nodelay = true })
local twist_subscriber     = nodehandle:subscribe("/vicon/headtwist",      'geometry_msgs/Twist',            10, { 'udp', 'tcp' }, { tcp_nodelay = true })


-- subscribe to vicon_receiver topic with 10 messages back-log
-- transport_options (arguments 4 & 5) are optional - used here only for demonstration purposes
if opt.publishTransform then 
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

if ros.ok() then
  ros.spinOnce()
  sys.sleep(0.1)
end

transform_subscriber:shutdown()
twist_subscriber:shutdown()

ros.shutdown()