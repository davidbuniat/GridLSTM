require 'rnn'
require 'model.GridLSTM_rnn'
require 'optim'


-- hyper-parameters 
batchSize = 8

input_size_x = 28
input_size_y = 28
input_k = 1
rnn_size = 64
output_size = 1
nIndex = 10
n_layers = 2
dropout = 0.5
should_tie_weights = 0
lr = 0.001
length = input_size_x* input_size_y
batch_size = 8
rho = length -- sequence length
local mnist = require 'mnist'

local trainset = mnist.traindataset()
local testset = mnist.testdataset()

print(trainset.size) -- to retrieve the size
print(testset.size) -- to retrieve the size

-- build simple recurrent neural network
local fwd = nn.GridLSTM(input_k, input_size_x, input_size_y, output_size, rnn_size, n_layers, rho, dropout, should_tie_weights)
local bwd = fwd:clone()
bwd:reset() -- reinitializes parameters
-- merges the output of one time-step of fwd and bwd rnns.
-- You could also try nn.AddTable(), nn.Identity(), etc.
local merge = nn.CAddTable() 
local brnn = nn.BiSequencer(fwd, bwd, merge)

-- internally, rnn will be wrapped into a Recursor to make it an AbstractRecurrent instance.
model = nn.Sequential()
model:add(brnn)
model:add(nn.JoinTable(1,1))
model:add(nn.Linear(rnn_size*length, 10))
model:add(nn.LogSoftMax())

print(model)

-- build criterion
criterion = nn.ClassNLLCriterion()--nn.SequencerCriterion(nn.ClassNLLCriterion())

-- this matrix records the current confusion across classes
classes = {'1','2','3','4','5','6','7','8','9','10'}
confusion = optim.ConfusionMatrix(classes)
--testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))


-- Load Batch
current_batch = 0
function next_batch(b_size)

   batch_x = torch.Tensor(b_size, 1, 28*28)
   batch_y = torch.Tensor(b_size)
   for i = 1, b_size do
       if(current_batch*b_size+i> trainset.size) then break end

       local ex = trainset[current_batch*(b_size)+i]
       local x = ex.x -- the input (a 28x28 ByteTensor)
       local y = ex.y -- the label (0--9) 

       batch_x[i] = ex.x:type('torch.DoubleTensor')
       batch_y[i] = ex.y+1
   end

   batch_x = prepro(batch_x)

   local inputs = {}
   local outputs = batch_y
   for i = 1, length do 
      table.insert(inputs, batch_x[i])
   end
   current_batch = current_batch + 1
   return inputs, outputs
end


-- preprocessing helper function
function prepro(x)
    x = x:permute(3,1,2):contiguous() -- swap the axes for faster indexing
    --y = y:permute(3,1,2):contiguous()
    return x
end



-- test function
function test(dataset)
   -- local vars
   local time = sys.clock()

   -- test over given dataset
   local c_batch = 1
   b_size = batch_size
   print('<trainer> on testing Set:')
   for t = 1,dataset.size,b_size do
      if(c_batch> 50) then break end
      batch_x = torch.Tensor(b_size, 1, 28*28)
      batch_y = torch.Tensor(b_size)
      for i = 1, b_size do
          if(c_batch*b_size+i> dataset.size) then break end
   
          local ex = dataset[c_batch*(b_size)+i]
          local x = ex.x -- the input (a 28x28 ByteTensor)
          local y = ex.y -- the label (0--9) 
   
          batch_x[i] = ex.x:type('torch.DoubleTensor')
          batch_y[i] = ex.y+1
      end
      c_batch = c_batch +1
      batch_x = prepro(batch_x)
   
      local inputs = {}
      local outputs = batch_y
      for i = 1, length do 
         table.insert(inputs, batch_x[i])
      end

      -- test samples
      local preds = model:forward(inputs)

      -- confusion:
      for i = 1,batch_size do
         confusion:add(preds[i], outputs[i])
      end
   end

   -- timing
   time = sys.clock() - time
   time = time / c_batch * b_size
   print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   print('% mean class accuracy (test set)'..confusion.totalValid * 100)
   confusion:zero()
end


-- training
local iteration = 1
local i = 0 
while true do
   -- 1. create a sequence of rho time-steps
   local inputs, targets = next_batch(batch_size)

   -- 2. forward sequence through rnn
   
   model:zeroGradParameters() 
   local outputs = model:forward(inputs)
   local err = criterion:forward(outputs, targets)
   
   print(string.format("Iteration %d ; NLL err = %f ", iteration, err))

   -- 3. backward sequence through rnn (i.e. backprop through time)
   
   local gradOutputs = criterion:backward(outputs, targets)
   local gradInputs = model:backward(inputs, gradOutputs)
   
   -- 4. update
   if(i==0) then 
      test(testset)
      
   end
   i = (i+1)%10
   model:updateParameters(lr)
   
   iteration = iteration + 1
end
