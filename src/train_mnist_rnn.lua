require 'rnn'
require 'model.gridLSTM_rnn'
require 'optim'
require 'util.SymmetricTable'

cmd = torch.CmdLine()
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-seed',123,'torch manual random number generator seed')

-- optimization
cmd:option('-learning_rate',0.02,'learning rate')
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')

opt = cmd:parse(arg)
torch.manualSeed(opt.seed)


-- hyper-parameters 

input_size_x = 14
input_size_y = 14
input_k = 4
rnn_size = 100
hiddenLayer = 4096
output_size = 1
nIndex = 10
n_layers = 1
dropout = 0.5 
should_tie_weights = 0
lr = opt.learning_rate
length = input_size_x * input_size_y
batch_size = 32
rho = length -- sequence length
local mnist = require 'mnist'

local trainset = mnist.traindataset()
local testset = mnist.testdataset()

print(trainset.size) -- to retrieve the size
print(testset.size) -- to retrieve the size

 
-- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
        print('using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
        print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end


-- Preprocessing of building blocks -- 

--- Create Modules
local grid_1 = nn.GridLSTM(input_k, input_size_x, input_size_y, output_size, rnn_size, n_layers, rho, dropout, should_tie_weights)
local grid_2 = grid_1:clone() -- Top-Right Corner
local grid_3 = grid_1:clone() -- Bottom-Left Corner
local grid_4 = grid_1:clone() -- Bottom-Right Corner

--- Reset clones
grid_2:reset()
grid_3:reset()
grid_4:reset()

--- Build GridLSTM 4 layers that process the data from different corners
local Seq_1 = nn.Sequencer(grid_1)   -- Top-Left Corner
local Seq_2 = nn.Sequencer(grid_2)   -- Top-Right Corner
local Seq_3 = nn.Sequencer(grid_3)   -- Bottom-Left Corner
local Seq_4 = nn.Sequencer(grid_4)   -- Bottom-Right Corner

--- grid_1 Top-Left Corner Stays the same
local SeqCorner_1  = Seq_1

--- grid_2 Top-Right Corner
local SeqCorner_2 = nn.Sequential()
SeqCorner_2:add(nn.SymmetricTable(input_size_x, input_size_y))   -- Symmetrify
SeqCorner_2:add(Seq_2)
SeqCorner_2:add(nn.SymmetricTable(input_size_x, input_size_y))   -- UnSymmetrify

--- grid_3 Bottom-Left Corner
local SeqCorner_3 = nn.Sequential()
SeqCorner_3:add(nn.SymmetricTable(input_size_x, input_size_y))   -- Symmetrify
SeqCorner_3:add(nn.ReverseTable())     -- Reverse
SeqCorner_3:add(Seq_3)
SeqCorner_3:add(nn.ReverseTable())     -- Unreverse
SeqCorner_3:add(nn.SymmetricTable(input_size_x, input_size_y))   -- UnSymmetrify

--- grid_4 Bottom-Right Corner
local SeqCorner_4 = nn.Sequential()
SeqCorner_4:add(nn.ReverseTable())     -- Reverse
SeqCorner_4:add(Seq_4)
SeqCorner_4:add(nn.ReverseTable())     -- Unreverse

--- Concat everything together
local concat = nn.ConcatTable()
concat:add(SeqCorner_1):add(SeqCorner_2):add(SeqCorner_3):add(SeqCorner_4)

--- Init Merger
-- Need to experiment with CMult
local merger = nn.Sequencer(nn.CAddTable())  

--- Final Merge
local gridLSTM = nn.Sequential()
gridLSTM:add(concat)
gridLSTM:add(nn.ZipTable())
gridLSTM:add(merger)

-- internally, rnn will be wrapped into a Recursor to make it an AbstractRecurrent instance.
model = nn.Sequential()
model:add(gridLSTM)
model:add(nn.JoinTable(1,1))
model:add(nn.Linear(rnn_size*length, hiddenLayer))
model:add(nn.Linear(hiddenLayer, 10))
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

       batch_x[i] = ex.x:type('torch.DoubleTensor')/255
       batch_y[i] = ex.y+1
   end

   batch_x = prepro(batch_x)

   local inputs = {}
   local outputs = batch_y
   local k = 1

   for y = 1, input_size_y do 
      for x = 1, input_size_x do
         k = 2*(y-1)*28 + 2*x-1
         table.insert(inputs, nn.JoinTable(2):forward{batch_x[k], batch_x[k+1], batch_x[k+input_size_x], batch_x[k+input_size_x+1]})
      end
   end

   --for i = 1, length do 
   --   table.insert(inputs, batch_x[k])
   --end

   current_batch = current_batch + 1
   return inputs, outputs
end


-- preprocessing helper function
function prepro(x)
   x = x:permute(3,1,2):contiguous() -- swap the axes for faster indexing
   if opt.gpuid >= 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        ---x = x:float():cuda()
   end
   return x
end



-- test function
function testing(dataset)
   -- local vars
   local time = sys.clock()

   -- test over given dataset
   local c_batch = 1
   b_size = batch_size
   print('<trainer> on testing Set:')
   for t = 1,dataset.size,b_size do
      if(c_batch*b_size> 500) then break end
      batch_x = torch.Tensor(b_size, 1, 28*28)
      batch_y = torch.Tensor(b_size)
      for i = 1, b_size do
          if(c_batch*b_size+i> dataset.size) then break end
   
          local ex = dataset[c_batch*(b_size)+i]
          local x = ex.x -- the input (a 28x28 ByteTensor)
          local y = ex.y -- the label (0--9) 
   
          batch_x[i] = ex.x:type('torch.DoubleTensor')/255
          batch_y[i] = ex.y+1
      end
      c_batch = c_batch +1
      batch_x = prepro(batch_x)
   
      local inputs = {}
      local outputs = batch_y
      local k = 1
      for y = 1, input_size_y do 
         for x = 1, input_size_x do
            k = 2*(y-1)*28 + 2*x-1
            table.insert(inputs, nn.JoinTable(2):forward{batch_x[k], batch_x[k+1], batch_x[k+input_size_x], batch_x[k+input_size_x+1]})
         end
      end

      --for i = 1, length do 
      --   table.insert(inputs, batch_x[i])
      --end

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

local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
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
      testing(testset)
      
   end
   i = (i+1)%25


   -- exponential learning rate decay
   if i == 0 and opt.learning_rate_decay < 1 then
      local decay_factor = opt.learning_rate_decay
      optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
      print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
      lr = optim_state.learningRate
   end

   model:updateParameters(lr)
   --model:forget() 
   
   iteration = iteration + 1
end
