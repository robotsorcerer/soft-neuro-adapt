--[[Train the network on glassfurnace data. User can train with 
    1) mlp (a simple feedforward network with one hidden layer)
    2) a recurrent network module stacked on a feedforward network
    3) a long short-term memory network

    Author: Olalekan Ogunmolu, December 2015 - May 2016
    Freely distributed under the MIT License.
  ]]
require 'torch'      
-- require 'optima.optim_'  
require 'data.dataparser'


function train_lstm(opt)
  local iter = iter or 0; 
  if opt.fastlstm then opt.rnnlearningRate = 5e-3 end
  local lr = opt.rnnlearningRate

  for t = 1, opt.maxIter do 
     -- 1. create a sequence of rho time-steps
    local inputs, targets = {}, {}
    inputs, targets = get_datapair(opt, t)

    --concatenate inputs along the columns to form a 1-D array
    inputs = torch.cat({inputs[1], inputs[2], inputs[3], inputs[4], 
                        inputs[5], inputs[6], inputs[7], inputs[8],
                        inputs[9]}, 2)
    -- print('inputs: \n', inputs)
    targets = torch.cat({targets[1], targets[2], targets[3]}, 2)

    --2. Forward sequence through rnn
    neunet:zeroGradParameters()
    neunet:forget()  --forget all past time steps

    local outputs = neunet:forward(inputs)

    local loss = cost:forward(outputs, targets) 
    if loss <=400 then lr = lr/(1+epoch) end
    
    if iter % 10  == 0 then collectgarbage() end

    --3. do backward propagation through time(Werbos, 1990, Rummelhart, 1986)
    local  gradOutputs  = cost:backward(outputs, targets)
    local gradInputs    = neunet:backward(inputs, gradOutputs) 

    --4. update lr
    neunet:updateParameters(lr)  
      
    if not opt.silent then 
      print(string.format("Epoch %d,iter = %d,  Loss = %.12f ", 
            epoch, iter, loss))
      if not opt.gru or opt.model=='mlp' then
        if(opt.weights) then
          print('neunet weights')
          print(neunet.modules[1].recurrentModule.modules[7].weight)

          print('neunet biases')
          print(neunet.modules[1].recurrentModule.modules[7].biases)
        end
      end
    end

    if opt.model=='lstm' then 
      if opt.gru then
        logger:add{['GRU training error vs. epoch'] = loss}
        logger:style{['GRU training error vs. epoch'] = '-'}
        if opt.plot then logger:plot()  end
      elseif opt.fastlstm then
        logger:add{loss}
        logger:style{['FastLSTM training error vs. epoch'] = '-'}
        if opt.plot then logger:plot()  end
      else
        logger:add{['LSTM training error vs. epoch'] = loss}
        logger:style{['LSTM training error vs. epoch'] = '-'}
        if opt.plot then logger:plot()   end
      end
    end
    
    collectgarbage()
    iter = iter +1 
  end  
end