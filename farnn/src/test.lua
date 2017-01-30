--[[ This file measures how generalizable the network model is.
      Author: Olalekan Ogunmolu. 
      Date:   Oct 20, 2016
      Lab Affiliation: Gans' Lab
      ]]
require 'torch'
require 'data.dataparser'
require 'rnn'

-- local optnet = require 'optnet'
torch.setdefaulttensortype('torch.FloatTensor')

--options and general settings -----------------------------------------------------
opt = {
  batchSize = 20,
  data = 'data',
  gpu = 0,
  noutputs = 1,
  display = 1,
  verbose = false,
  maxIter = 100,
  hiddenSize = {2, 5, 10},
  backend    = 'cudnn',
  checkpoint = 'network/data_lstm-net.t7'
}

for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

if opt.display==0 then opt.display = false end

local function init()

  --GPU Settings -----------------------------------------------------------------------
  local use_cuda = false

  local msg
  if opt.gpu >= 0  and opt.backend == 'cudnn' then
    require 'cunn'
    require 'cutorch'
    require 'cudnn'
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


--test function-----------------------------------------------------------------
local function test(opt, model)
 -- local vars
 local splitData = {}; 
 splitData = split_data(opt)
 local time = sys.clock()
 local testHeight = splitData.test_input[1]:size(1)
 -- averaged param use?
 if average then
    cachedparams = parameters:clone()
    parameters:copy(average)
 end
 -- test over given dataset
 print('<trainer> on testing Set:')

  local preds;
  local iter = 0;  

  -- create mini batch        
  local inputs, targets = {}, {}      

  for t = 1, math.min(opt.maxIter, testHeight) do    
    if(t >= opt.maxIter) then t = 1 end  --wrap around
     inputs, targets = get_datapair(opt, t)
    if opt.noutputs  == 1 then 
      --concat tensors 1 and 4 (in and pitch along 2nd dimension)
      inputs = torch.cat({inputs[1], inputs[4]}, 2) 
      -- target would be expected to be a tensor rather than a table since we are using sequencer
      targets = targets[3]
    else
      inputs = torch.cat({inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], inputs[7]}, 2)  
      targets = transfer_data( torch.cat({targets[1], targets[2], targets[3], targets[4], targets[5], targets[6]}, 2) )
    end

    if opt.gpu >= 0 then inputs:cuda(); targets:cuda() end
    inputs= inputs:cuda(); targets = targets:cuda()

    -- test samples
    model:forget()  --forget all past time steps
    local preds = model:forward(inputs)

    -- not necessary[[
    require 'rnn'
    local cost = nn.SequencerCriterion(nn.MSECriterion()):cuda()
    local loss = cost:forward(preds, targets) 

    if iter % 10  == 0 then collectgarbage() end

    --]]
     
    -- timing
    time = sys.clock() - time
    time = time / testHeight

    -- if  (iter*opt.batchSize >= math.min(opt.maxIter, testHeight))  then 
      print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')  
      -- if not (opt.data=='glassfurnace') then print("avg. prediction errors on test data", normedPreds) 

        print(inputs)
        print(string.format('actual %s preds: %s', tostring(targets), tostring(preds)))
        -- sys.sleep(3)
        print(string.format("iter = %d,  Loss = %f ",  iter, loss))
      -- else print("avg. prediction errors on test data", avg/normedPreds) end
    -- end 
    iter = iter + 1
  end  
end

local function connect_cb(name, topic)
  print("subscriber connected: " .. name .. " (topic: '" .. topic .. "')")
end

local function disconnect_cb(name, topic)
  print("subscriber diconnected: " .. name .. " (topic: '" .. topic .. "')")
end

local function main()  
  local model = init()
  for i=1,50 do
    test(opt, model)
  end
end

main()