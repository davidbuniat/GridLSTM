-- The problem

require 'rnn'
require 'model.GridLSTM_1D'
require 'optim'
require 'util.SymmetricTable'

cmd = torch.CmdLine()
cmd:option('-gpuid',-1,'which gpu to use. -1 = use CPU')
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
input_size_t = 49
rnn_size = 400
output_size = 1
n_layers = 1
dropout = 0
should_tie_weights = 0
lr = opt.learning_rate
length = input_size_t
batch_size = 15
rho = 49 -- sequence length

nngraph.setDebug(false)


-- Build 15 digit addition problem
local input = nn.ConcatTable()
input:add(nn.Identity())
input:add(nn.Identity())

local connect = nn.ParallelTable()
connect:add(nn.Linear(rnn_size, 1))
connect:add(nn.Linear(rnn_size, 1))


local gridLSTM = nn.Sequential()
--gridLSTM:add(nn.LSTM(1, rnn_size))
--gridLSTM:add(nn.LSTM(rnn_size, rnn_size))
gridLSTM:add(input)
gridLSTM:add(nn.GridLSTM_1D(1, rnn_size, n_layers, rho, should_tie_weights, dropout))
gridLSTM:add(nn.Linear(1, 11))
gridLSTM:add(nn.LogSoftMax())

model = nn.Sequential()
model:add(nn.Sequencer(gridLSTM))
--model:add(nn.JoinTable(1,1))

criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

print(model)
torch.manualSeed(123)

-- preprocessing helper function
function prepro(x)
   x = x:permute(3,1,2):contiguous() -- swap the axes for faster indexing
   if opt.gpuid >= 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        ---x = x:float():cuda()
   end
   return x
end

local function next_batch()
	b_size = batch_size

	local x = torch.DoubleTensor(b_size,	1, 49):zero()
	local y = torch.IntTensor(b_size,	1, 49):zero()

	x = x-1 --init to -1
	y = y-1 --init to -1
	
	for ind = 1, b_size do

		for i = 2,15 do
			x[ind][1][i] = torch.random(0, 9)

		end 

		for i = 17,31 do
			x[ind][1][i] = torch.random(0, 9)

		end 
	end

	for ind = 1, b_size do 
		local mem = 0
		for i = 14,1,-1 do

			y[ind][1][i+33] = math.floor(x[ind][1][i+1] + x[ind][1][i+16] + mem)

			if y[ind][1][i+33] > 9 then
				y[ind][1][i+33] = y[ind][1][i+33]-10
				mem = 1
			else
				mem = 0
			end
		end
		
		y[ind][1][33] = mem
	end
	
	x = prepro(x) 
	y = prepro(y) + 2

	inputs = {}
	outputs = {}
	for i = 1, length do

		table.insert(inputs, x[i])
		table.insert(outputs,y[i]:view(y[i]:nElement()))
		
	end 

	return inputs,outputs
end

local function testing()
	local mean_acc = 0 
	for k = 1, 7 do 

		inputs, outputs = next_batch(batch_size)
		local preds = model:forward(inputs)
		--print(outputs)
		local acc = 0
		for ind = 33, length-1 do
			y, i = torch.max(preds[ind],2)
			for j = 1, batch_size do

				if outputs[ind][j] == i[j][1] then
					acc = acc + 1
				end
			end
		end

		acc = acc / (batch_size * 16)

		mean_acc = mean_acc + acc
	end
	mean_acc = 100*mean_acc / 7
	print('accuracy is '..mean_acc..'%')
end

x_weights, dl_dx = model:getParameters()

feval = function(x_new)
    -- copy the weight if are changed
    if x_weights ~= x_new then
        x_weights:copy(x_new)
    end
    -- select a training batch
    local inputs, targets = next_batch(batch_size)
    -- reset gradients (gradients are always accumulated, to accommodate
    -- batch methods)
    dl_dx:zero()

    -- evaluate the loss function and its derivative with respect to x_weights, given a mini batch
    local prediction = model:forward(inputs)
    local loss_x = criterion:forward(prediction, targets)
    model:backward(inputs, criterion:backward(prediction, targets))

    return loss_x, dl_dx
end

adam_params = {
   learningRate = lr,
   --learningRateDecay = 1e-4,
   --weightDecay = 0,
   --momentum = 0
}


local iteration = 1
local i = 0 
while true do

	_, fs = optim.adam(feval,x_weights,adam_params)
   ---- 1. create a sequence of rho time-steps
   --local inputs, targets = next_batch(batch_size)
   ---- 2. forward sequence through rnn
   --model:zeroGradParameters()
   -- 
   --local outputs = model:forward(inputs) 
   ----print(targets)
   --local err = criterion:forward(outputs, targets)
   
   print(string.format("Iteration %d ; NLL err = %f ", iteration, fs[1]))

   -- 3. backward sequence through rnn (i.e. backprop through time)
   
   --local gradOutputs = criterion:backward(outputs, targets)
   --local gradInputs = model:backward(inputs, gradOutputs)
   
   -- 4. update
   if(i==0) then 
      testing()
      
   end
   i = (i+1)%25

   --model:updateParameters(lr)
   --model:forget() 
   
   iteration = iteration + 1
end
