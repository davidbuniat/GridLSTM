require 'rnn'
require 'model.gridLSTM_rnn'
require 'optim'
require 'util.SymmetricTable'
require 'image'


cmd = torch.CmdLine()
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-seed',123,'torch manual random number generator seed')

-- optimization
cmd:option('-learning_rate',0.001, ' learning rate')
cmd:option('-learning_rate_decay',1,'learning rate decay')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')

opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
nngraph.setDebug(false)

-- hyper-parameters 
p_size = 1
input_size_x = 30--2--36
input_size_y = 23--5--27
n_labels = 1
input_k = p_size * p_size * 3
rnn_size = 10
hiddenLayer = 40
output_size = 9
nIndex = 10
n_layers = 1
dropout = 0
should_tie_weights = 0
lr = opt.learning_rate
length = input_size_x * input_size_y
batch_size = 16
rho = length -- sequence length
load = false
n_f_hidden = 2048 -- number of final hidden layers
depth_scale_factor = 5.5


--------------------------------------------------------------
--                          Data
--------------------------------------------------------------

local nyu = require 'data.nyu'

local trainset = nyu.traindataset
local testset = nyu.testdataset

img_x = trainset.x:size()[4]
img_y = trainset.x:size()[3]
print(img_x)
print(img_y)

print(trainset.size) -- to retrieve the size
print(testset.size) -- to retrieve the size

--------------------------------------------------------------
--                          GPU
--------------------------------------------------------------

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

zeroTensor = torch.Tensor()
if opt.gpuid >= 0 then 
  zeroTensor = torch.CudaTensor()
end

--------------------------------------------------------------
--                         GridLSTM
--------------------------------------------------------------

-- Preprocessing of building blocks
local input = nn.ConcatTable()
input:add(nn.Linear(input_k, rnn_size))
input:add(nn.Linear(input_k, rnn_size))

----- Create Modules
local grid_1 = nn.Sequential()
grid_1:add(input)
grid_1:add(nn.GridLSTM(input_size_x, input_size_y, rnn_size, rho, should_tie_weights, zeroTensor))

local grid_2 = nn.GridLSTM(input_size_x, input_size_y, rnn_size, rho, should_tie_weights, zeroTensor) -- template_hidden -- Top-Right Corner
local grid_3 = nn.GridLSTM(input_size_x, input_size_y, rnn_size, rho, should_tie_weights, zeroTensor)-- Bottom-Left Corner

local grid_4 = nn.Sequential() -- Bottom-Right Corner
grid_4:add(nn.GridLSTM(input_size_x, input_size_y, rnn_size, rho, should_tie_weights, zeroTensor))
grid_4:add(nn.JoinTable(1,1))

--- Build GridLSTM 4 layers that process the data from different corners
local grid_1 = nn.Sequencer(grid_1)   -- Top-Left Corner
local grid_2 = nn.Sequencer(grid_2)   -- Top-Right Corner
local grid_3 = nn.Sequencer(grid_3)   -- Bottom-Left Corner
local grid_4 = nn.Sequencer(grid_4)   -- Bottom-Right Corner

--- grid_1 Top-Left Corner Stays the same
local SeqCorner_1  = grid_1

--- grid_2 Top-Right Corner
local SeqCorner_2 = nn.Sequential()
SeqCorner_2:add(nn.SymmetricTable(input_size_x, input_size_y))   -- Symmetrify
SeqCorner_2:add(grid_2)
SeqCorner_2:add(nn.SymmetricTable(input_size_x, input_size_y))   -- --UnSymmetrify

--- grid_3 Bottom-Left Corner
local SeqCorner_3 = nn.Sequential()
SeqCorner_3:add(nn.SymmetricTable(input_size_x, input_size_y))   -- Symmetrify
SeqCorner_3:add(nn.ReverseTable())     -- Reverse
SeqCorner_3:add(grid_3)
SeqCorner_3:add(nn.ReverseTable())     -- Unreverse
SeqCorner_3:add(nn.SymmetricTable(input_size_x, input_size_y))   -- --UnSymmetrify

--- grid_4 Bottom-Right Corner
local SeqCorner_4 = nn.Sequential()
SeqCorner_4:add(nn.ReverseTable())     -- Reverse
SeqCorner_4:add(grid_4)
SeqCorner_4:add(nn.ReverseTable())     -- Unreverse

local finalLayer = nn.Sequential()
finalLayer:add(nn.Linear(2*rnn_size, n_f_hidden))
finalLayer:add(nn.ReLU())
finalLayer:add(nn.Linear(n_f_hidden, n_labels*p_size*p_size))

--- Init Merger
local merger = nn.Sequencer(nn.CAddTable(1,1))  

--- Final Merge of for concurrent layers
local gridLSTM = nn.Sequential()

gridLSTM:add(SeqCorner_1)
gridLSTM:add(SeqCorner_2)
gridLSTM:add(SeqCorner_3)
gridLSTM:add(SeqCorner_4)
gridLSTM:add(nn.Sequencer(finalLayer))

local model = gridLSTM
if load then model = torch.load('gridlstm.model') end 

crit = nn.MSECriterion()
criterion = nn.SequencerCriterion(crit)

if opt.gpuid >= 0 then
  model:cuda()
  criterion:cuda()
end

--------------------------------------------------------------
--                     Batch Manipulation
--------------------------------------------------------------
-- Load Batch
function get_batch(dataset, n_batch)
  -- Create tensors
  batch_x = torch.Tensor(batch_size, 3, img_x*img_y)
  batch_y = torch.Tensor(batch_size, 1, img_x*img_y)

  -- Load data
  for i = 1, batch_size do
      if((n_batch-1)*(batch_size)+i> dataset.size) then break end

      local x = dataset.x[(n_batch-1)*(batch_size)+i]
      local y = dataset.y[(n_batch-1)*(batch_size)+i]
  
      batch_x[i] = x
      batch_y[i] = y
  end
  return batch_x, batch_y
end

-- Preprocessing helper function
function prepro(x)

   x = x:permute(3,1,2):contiguous() -- swap the axes for faster indexing
   if opt.gpuid >= 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        x = x:float():cuda()
   end

   return x
end

function patchify(batch_x, batch_y)
  inputs =  nn.SplitTable(1):forward(batch_x)
  outputs  = nn.SplitTable(1):forward(batch_y)
  inputs = nn.NarrowTable(1,input_size_x*input_size_y):forward(inputs)
  outputs = nn.NarrowTable(1,input_size_x*input_size_y):forward(outputs)
--
--  local inputs = {}
--  local outputs = {}
--  local k = 1
--
--  for y = 1, input_size_y do 
--    for x = 1, input_size_x do
--      k = p_size*(y-1)*input_size_x + p_size*(x-1) + 1
--
--      local patch_x  = {}
--      local patch_y  = {}
--
--      -- Get Patch with p_size
--      for x_p = 0, p_size-1 do 
--        for y_p = 0, p_size-1 do
--
--          table.insert(patch_x, batch_x[k+y_p*p_size+x_p])
--          table.insert(patch_y, batch_y[k+y_p*p_size+x_p]) -- :squeeze()
--        end 
--      end 
--
--      local joining = nn.JoinTable(2)   
--      if opt.gpuid >= 0 then
--        joining:cuda()
--      end
--
--      table.insert(inputs, joining:forward{unpack(patch_x)})
--      table.insert(outputs, unpack(patch_y))
--    end
--  end

  return inputs, outputs
end

function next_batch(dataset, c_batch)

  batch_x, batch_y = get_batch(dataset, current_batch)
  
  batch_x = prepro(batch_x)
  batch_y = prepro(batch_y)

  inputs, outputs = patchify(batch_x, batch_y)

  return inputs, outputs
end

--------------------------------------------------------------
--                        Test Function
--------------------------------------------------------------

function testing(dataset)
   -- local vars
   local time = sys.clock()
   b_size = batch_size
   local testset = {}
   local outputset = {}

   -- test over given dataset
   local c_batch = 1
   rms_sum = 0
   print('<trainer> on testing Set:')
   for t = 1,dataset.size,b_size do
      if(c_batch*b_size> 500) then break end

      inputs, outputs = next_batch(dataset, c_batch)

      -- Test samples
      local preds = model:forward(inputs)
      outputs = nn.JoinTable(2):forward{unpack(outputs)}
      preds = nn.JoinTable(2):forward{unpack(preds)}

      t_preds = preds:reshape(b_size, input_size_y, input_size_x)
      t_output = outputs:reshape(b_size, input_size_y, input_size_x)

      for i = 1, batch_size do
        table.insert(testset, t_preds[i])
        table.insert(outputset, t_output[i])
      end

      -- Scale by 5.5 to get initial results
      preds = preds + 0.5
      outputs = outputs + 0.5
      preds:mul(depth_scale_factor)
      outputs:mul(depth_scale_factor)

      -- Calculate Root Mean Square Error
      for i = 1, batch_size do
        local rms = torch.sqrt(torch.sum(torch.pow(torch.csub(preds[i],outputs[i]),2))/((input_size_x*input_size_y)))
        rms_sum= rms_sum + rms
      end

      c_batch = c_batch + 1
   end

   rms = rms_sum/(torch.floor(dataset.size/b_size)*b_size)
   print("Root mean squared error: " .. rms )
   matio.save('data/testset.mat', t_preds)
   matio.save('data/outputset.mat', t_output)
   -- timing
   time = sys.clock() - time
   time = time / (c_batch * b_size)
   print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

end

--------------------------------------------------------------
--                      Training
--------------------------------------------------------------

x_weights, dl_dx = model:getParameters()
current_batch = 1

feval = function(x_new)
    -- copy the weight if are changed
    if x_weights ~= x_new then
        x_weights:copy(x_new)
    end

    -- select a training batch
    local inputs, targets = next_batch(trainset, current_batch)
    -- reset gradients (gradients are always accumulated, to accommodate
    -- batch methods)
    dl_dx:zero()

    -- evaluate the loss function and its derivative with respect to x_weights, given a mini batch
    local prediction = model:forward(inputs)
    --print(prediction[1])
    --print(targets[1])
    local loss_x = criterion:forward(prediction, targets)

    model:backward(inputs, criterion:backward(prediction, targets))

    current_batch = current_batch + 1 -- iterate
    return loss_x, dl_dx
end

adam_params = {
   learningRate = lr,
   --learningRateDecay = 1e-4,
   --weightDecay = 0,
   --momentum = 0
}

local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
-- training
local iteration = 1
local i = 0
while true do

    _, fs = optim.adam(feval,x_weights,adam_params)
   print(string.format("Iteration %d ; NLL err = %f ", iteration, fs[1]))
      
   if(i==0 and iteration<160) then
      testing(testset)
      --print('error for iteration ' .. sgd_params.evalCounter  .. ' is ' .. fs[1] / rho)
   end
   i = (i+1)%10

   iteration = iteration + 1
end


--------------------------------------------------------------
--                      testing
--------------------------------------------------------------

--batch_x, batch_y = next_batch(testset, 1)
----
--batch_y = nn.JoinTable(2):forward{unpack(batch_y)}
--batch_y = batch_y:reshape(batch_size, input_size_y, input_size_x)
--a = testset.y[2]:squeeze()
--b = batch_y[2]
--
--matio.save('data/testset.mat', batch_y)
--matio.save('data/outputset.mat', testset.y)

