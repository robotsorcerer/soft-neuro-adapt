--[[
  Author: Olalekan Ogunmolu, December 2015 - May 2016
  Freely distributed under the MIT License
]]
require 'torch'
local matio = require 'matio'

function transfer_data(x)
  if use_cuda then
    return x:cuda()
  else
    return x:double()
  end
end

local function data_path_printer(x)  
  print(sys.COLORS.green .. string.format("you have specified the data path %s", x))
end

local function get_filename(x)
	local filename = x:match("(%a+)")
	local filenamefull = 'data/' .. filename .. '.mat'
  return filename, filenamefull
end

function split_data(opt)	
	local filename, filenamefull = get_filename(opt.data)  -- we strip the filename extension from the data
	  -- if epoch==1 then data_path_printer(filenamefull) ends
	  local splitData = {}
	--ballbeam and robotarm are siso systems from the DaiSy dataset
	if(string.find(filename, 'data')) then 
		--[[
		data consists of the following fields in order:
		'base_in', 'base_out', 'left_in', 'left_out', ...
           'right_in', 'right_out', 'x', 'y', 'z', 'pitch', 'yaw'
		]]
	  data = matio.load(filenamefull);  
	  input = {data.left_in, data.left_out, data.right_in, data.right_out,
				data.base_in, data.base_out}
	  out   = { 
	                  data.z,       -- z
	                  data.pitch,   -- pitch
	                  data.yaw		-- yaw
	            }

	  k           = input[1]:size(1)    
	  off         = torch.ceil( torch.abs(0.8*k))

	  splitData.train_input = { 
	  							input[1][{{1, off}, {1}}], input[2][{{1, off}, {1}}],  									-- order must be preserved. cuda tensor does not support csub yet
	  							input[3][{{1, off}, {1}}], input[4][{{1, off}, {1}}],           -- most of the work is done here              (out[{{1, off}, {1}}])/10, outlln[{{1, off}, {1}}], 
	  							input[5][{{1, off}, {1}}], input[6][{{1, off}, {1}}]
	  						  }

	  splitData.train_out   = { 
	                 			out[1][{{1, off}, {1}}], out[2][{{1, off}, {1}}],            -- most of the work is done here              (out[{{1, off}, {1}}])/10, outlln[{{1, off}, {1}}], 
	                 			out[3][{{1, off}, {1}}]
	                		  } 

	  -- splitData.train =  data[{{1, off}, {1, 7}}] 
	  --create testing data
	  -- splitData.test =  data[{{off + 1, k}, {1, 7}}]
	  splitData.test_input = {
	  							input[1][{ {off+1, k}, {1}}], input[2][{ {off+1, k}, {1}}],
	  							input[3][{ {off+1, k}, {1}}], input[4][{ {off+1, k}, {1}}],
	  							input[5][{ {off+1, k}, {1}}], input[6][{ {off+1, k}, {1}}]
	  						  }

	  splitData.test_out   = {
				                 out[1][{{off+1, k}, {1}}], out[2][{{off+1, k}, {1}}], 
				                 out[3][{{off+1, k}, {1}}]
			                 }  

	  width       = splitData.train_input[1]:size(2)
	  height      = splitData.train_input[1]:size(1)

	  ninputs     = opt.ninputs; noutputs = opt.noutputs; 
	  nhiddens, nhiddens_rnn = opt.hiddenSize[1], opt.hiddenSize[1]; 
	  
	-- MIMO dataset from the Daisy  glassfurnace dataset (3 inputs, 6 outputs)
	elseif (string.find(filename, 'glassfurnace')) then
	  data = matio.load(filenamefull)  ;   data = data[filename];
	  -- three inputs i.e. heating input, cooling input, & heating input
	  input =   data[{{}, {2, 4}}]   

	  -- six outputs from temperatures sensors in a cross sectiobn of the furnace          
	  out =   data[{{}, {5,10}}] 

	  k = input:size(1)
	  off = torch.ceil(torch.abs(0.6*k))
	  --create actual training datasets
	  	splitData.train_input = input[{{1, off}, {}}];
	  	splitData.train_out   =   out[{{1, off}, {}}];
	  	splitData.test_input  = input[{{off+1, k}, {}}];	  	splitData.test_out	  = out[{{off+1, k}, {}}];

	  width       = splitData.train_input:size(2)
	  height      = splitData.train_input:size(2)
	  ninputs     = opt.ninputs; 
	  -- if (ninputs == 2) then noutputs = 1 else  noutputs = 6 end
	  noutputs = opt.noutputs
	  nhiddens = 6; nhiddens_rnn = 6 
	end
	return splitData
end

function batchNorm(x, N)
  --apply batch normalization to training data 
  -- http://arxiv.org/pdf/1502.03167v3.pdf
  local eps = 1e-5
  local momentum = 1e-1
  local affine = true
  local BN = nn.BatchNormalization(N, eps, momentum, affine)

  if type(x) == 'userdata' then       --inputs
   if opt.batchNorm then x  = BN:forward(x) end
    x  = transfer_data(x) --forward doubleTensor as CudaTensor
  elseif type(x) == 'table' then
    for i = 1, #x do
      if opt.batchNorm then  x[i] = BN:forward(x[i]) end
      x[i] = transfer_data(x[i])
    end
  end  
  collectgarbage()    
  return x
end

local optlocal = {
data = 'data',
ninputs = 9,
noutputs = 3,
hiddenSize = {9, 3, 5}
}

splitData = {}
splitData = split_data(optlocal)	

function get_datapair(opt, stage)	
	local inputs, targets = {}, {}
	local test_inputs, test_targets = {}, {}

	local testHeight = splitData.test_out[1]:size(1)
	if (opt.data=='data') then  
	 -- 1. create a sequence of rho time-steps
	 --input is a sequence of past outputs  delayed by a 5 steps
	 --and current input
		train_inputs 		= {
						splitData.train_input[1]:narrow(1, stage+1, opt.batchSize),    															  		-- >  u(t)
						splitData.train_input[2]:narrow(1, stage+1, opt.batchSize), 
						splitData.train_input[3]:narrow(1, stage+1, opt.batchSize),   		-- }
					   	splitData.train_input[4]:narrow(1, stage+1, opt.batchSize), 
					   	splitData.train_input[5]:narrow(1, stage+1, opt.batchSize),   		-- }  -->y_{t-1}
					   	splitData.train_input[6]:narrow(1, stage+1, opt.batchSize),		-- }
					   	splitData.train_out[1]:narrow(1, stage, opt.batchSize), 
						splitData.train_out[2]:narrow(1, stage, opt.batchSize), 				--}	
		                splitData.train_out[3]:narrow(1, stage, opt.batchSize)
					  } 
		-- batch of train targets
		train_targets 	 = {  
						  splitData.train_out[1]:narrow(1, stage+1, opt.batchSize), 
						  splitData.train_out[2]:narrow(1, stage+1, opt.batchSize), 				--}	
		                  splitData.train_out[3]:narrow(1, stage+1, opt.batchSize)
		                }
		-- test inputs
		test_inputs = {
						splitData.test_input[1]:narrow(1, stage+1, opt.batchSize), 																  		-- u(t)
						splitData.test_input[2]:narrow(1, stage+1, opt.batchSize), 
						splitData.test_input[3]:narrow(1, stage+1, opt.batchSize),	  		-- 
						splitData.test_input[4]:narrow(1, stage+1, opt.batchSize), 
						splitData.test_input[5]:narrow(1, stage+1, opt.batchSize),     		-- } y_{t-1}
						splitData.test_input[6]:narrow(1, stage+1, opt.batchSize),
						 splitData.test_out[1]:narrow(1, stage, opt.batchSize), 
						 splitData.test_out[2]:narrow(1, stage, opt.batchSize),		--}
		                 splitData.test_out[3]:narrow(1, stage, opt.batchSize)
					  }		
		--test targets
		test_targets = {
						 splitData.test_out[1]:narrow(1, stage+1, opt.batchSize), 
						 splitData.test_out[2]:narrow(1, stage+1, opt.batchSize),		--}
		                 splitData.test_out[3]:narrow(1, stage+1, opt.batchSize)
		             	}

		--pre-whiten the inputs and outputs in the mini-batch
		local N = 1
		train_inputs 	= batchNorm(train_inputs, N)
		train_targets = batchNorm(train_targets, N)

		test_inputs = batchNorm(test_inputs, N)
		test_targets = batchNorm(test_targets, N)

	elseif (opt.data == 'glassfurnace') then   --MIMO Dataset
		offsets = torch.LongTensor(opt.batchSize):random(1,height)  
		test_offsets = torch.LongTensor(opt.batchSize):random(1,testHeight) 
		
		-- print('train_inputs', splitData.train_input)
		--recurse inputs and targets into one long sequence
		train_inputs = 	splitData.train_input:index(1, offsets)		
		test_inputs =  splitData.test_input:index(1, test_offsets) 

		--batch of targets
		train_targets = 	splitData.train_out:index(1, offsets)        
		test_targets =   splitData.test_out:index(1, test_offsets)  
		                  
		--pre-whiten the inputs and outputs in the mini-batch
		train_inputs = batchNorm(train_inputs, 3)
		train_targets = batchNorm(train_targets, 6)

		test_inputs = batchNorm(test_inputs, 3)		
		test_targets = batchNorm(test_targets, 6)

		--increase offsets indices by 1      
		offsets:add(1) -- increase indices by 1
		test_offsets:add(1)
		offsets[offsets:gt(height)] = 1  
		test_offsets[test_offsets:gt(testHeight)] = 1
	end
return train_inputs, train_targets, test_inputs, test_targets
end