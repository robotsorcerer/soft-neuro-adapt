local ros = require 'ros'
require('sys')
ros.init('service_client_demo')

local nh = ros.NodeHandle()

local clientB = ros.ServiceClient('/mannequine_head/predictor_params', 'nn_controller/predictor_params')

print('Service spec:')
print(clientB.spec)

-- we can either create the request message explicitely
local req_msg = clientB:createRequest()
-- req_msg:fillFromTable({logger="my_dummy_logger", level="warn"})

print('Request:')
print(req_msg)

print('Calling service: ' .. clientB:getService())

-- call the service
while(ros.ok()) do 
	response = clientB:call(req_msg)	
	sys.sleep('0.5')
	print(response)
	print('\t\n\t\t|\t\n\t\t|\t\n\t\tv\t\n\t\tResponse')
end

-- print(response)


ros.shutdown()
