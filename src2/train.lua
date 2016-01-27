--This code is based on https://github.com/coreylynch/grid-lstm by Corey Lynch

require 'nn'
require 'nngraph'
require 'optim'
require 'util.OneHot'
require 'util.misc'
require 'rnn'

require 'torch'
require 'lfs'
--require 'cudnn'

nngraph.setDebug(false)

local data = require 'data.afreight'
local GridLstm = require 'model.GridLSTM'
local model_utils = require 'util.model_utils'

cmd = torch.CmdLine()
cmd:text('Train a 1 dimensional grid lstm')
cmd:text('Options')
cmd:option('-input_k', 3, 'size of input bit vector')
cmd:option('-n', 4, 'number of layers')
cmd:option('-mb', 8, 'minibatch size')
cmd:option('-iters', 100000, 'number of iterations')

-- input params
cmd:option('-n_data', 16, 'Number of the data')
cmd:option('-n_x', 4, 'width of the image')
cmd:option('-n_y', 4, 'height of the image')

cmd:option('-width', 160, 'length of the image')
cmd:option('-height', 120, 'height of the image ')

-- model params
cmd:option('-rnn_size', 32, 'size of LSTM internal state')
cmd:option('-num_layers', 1, 'number of layers in the LSTM')
cmd:option('-model', 'grid_lstm', 'lstm, grid_lstm, gru, or rnn')
cmd:option('-tie_weights', 0, 'tie grid lstm weights?')

-- optimization
cmd:option('-learning_rate',2e-3,'learning rate')
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-learning_rate_decay_after',10,'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')

cmd:option('-dropout',0,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-seq_length',256,'number of timesteps to unroll for') -- 19200
cmd:option('-batch_size',8,'number of sequences to train on in parallel')
cmd:option('-max_epochs',50,'number of full passes through the training data')

-- bookkeeping
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-print_every',1,'how many steps/minibatches between printing out the loss')
cmd:option('-eval_val_every',1000,'every how many iterations should we evaluate on validation data?')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-savefile','lstm','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
cmd:option('-accurate_gpu_timing',0,'set this flag to 1 to get precise timings when using GPU. Might make code bit slower but reports accurate timings.')


cmd:option('-grad_clip',5,'clip gradients at this value')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

opt.seq_length = opt.n_x * opt.n_y;

local loader =  data.create(opt.n_data, opt.width, opt.height, opt.batch_size) 

if opt.gpuid > 0 then
    cunn = require 'cunn'
    cutorch = require 'cutorch'
    cutorch.setDevice(opt.gpu)
    cutorch.manualSeed(opt.seed)
end

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

print(sys.COLORS.red ..  '==> Bulding the Model')



-- define the model: prototypes for one timestep, then clone them in time
print('creating an Grid LSTM with ' .. opt.num_layers .. ' layers')
protos = {}
protos.rnn = GridLstm.grid_lstm(opt.input_k, loader.labels_size, opt.rnn_size, opt.num_layers, opt.dropout, opt.tie_weights)
protos.criterion = nn.ClassNLLCriterion()

print(sys.COLORS.red ..  '==> Bulding the Model 1')

-- the initial state of the cell/hidden states
init_state = {}
for L=1,opt.num_layers do
    local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
    if opt.gpuid >=0 then h_init = h_init:cuda() end
    table.insert(init_state, h_init:clone())
    table.insert(init_state, h_init:clone()) -- extra initial state for prev_c
    -- Need Extra states for y,x dimensions
end

-- ship the model to the GPU if desired
if opt.gpuid >= 0 then
    for k,v in pairs(protos) do v:cuda() end
end

-- put the above things into one flattened parameters tensor
params, grad_params = model_utils.combine_all_parameters(protos.rnn)


-- initialization
if do_random_init then
    params:uniform(-0.08, 0.08) -- small uniform numbers
end

-- initialize the LSTM forget gates with slightly higher biases to encourage remembering in the beginning
for layer_idx = 1, opt.num_layers do
    for _,node in ipairs(protos.rnn.forwardnodes) do
        if node.data.annotations.name == "i2h_" .. layer_idx then
            print('setting forget gate biases to 1 in LSTM layer ' .. layer_idx)
            -- the gates are, in order, i,f,o,g, so f is the 2nd block of weights
            node.data.module.bias[{{opt.rnn_size+1, 2*opt.rnn_size}}]:fill(1.0)
        end
    end
end

print('number of parameters in the model: ' .. params:nElement())

local timer = torch.Timer()
-- make a bunch of clones after flattening, as that reallocates memory
clones = {}
for name,proto in pairs(protos) do
    print('cloning ' .. name)
    clones[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters)
end
local time = timer:time().real
print('Passed: '..time)

-- preprocessing helper function
function prepro(x,y)
    x = x:permute(3,1,2):contiguous() -- swap the axes for faster indexing
    y = y:permute(3,1,2):contiguous()
    if opt.gpuid >= 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        x = x:float():cuda()
        y = y:float():cuda()
    end
    return x,y
end

-- do fwd/bwd and return loss, grad_params
local init_state_global = clone_list(init_state)
function feval(x)
    if x ~= params then
        params:copy(x)
    end

    ------------------ get minibatch -------------------
    local x, y = loader:next_batch(1)
  	x_input,y = prepro(x,y)

    ------------------- forward pass -------------------
    local rnn_state = {[0] = init_state_global}
    -- Copy along previous row 
    for i = 0,opt.n_x do
    	rnn_state[-i] = init_state_global
    end

    
    local predictions = {}           -- softmax outputs
    local loss = 0

    -- Iterate for each coordinate
    for x=1,opt.n_x do
    	for y=1,opt.n_y do
    		-- Get coordinates
    		local xy = (x-1)*opt.n_x+y 			-- x,y coordinate in 1D
    		local prev_x = (x-2)*(opt.n_x)+y 	-- x-1,y coordinate in 1D
    		local prev_y = (x-1)*opt.n_x+y-1	-- x,y-1 coordinate in 1D

        	clones.rnn[xy]:training() -- make sure we are in correct mode (this is cheap, sets flag)
        	local rnn_inputs
        	local input_mem_cell = torch.zeros(opt.batch_size, opt.rnn_size)
        	if opt.gpuid >= 0 then
        		input_mem_cell = input_mem_cell:float():cuda()
        	end
        	-- Concat states of previous dimension x and y into prev_state
        	prev_x = {unpack(rnn_state[prev_x])}
        	prev_state = {unpack(rnn_state[prev_y])} 				-- include y dimensions
        		for k,v in pairs(prev_x) do prev_state[k+#prev_x] = v end -- copy from prev_x

        	rnn_inputs = {input_mem_cell, x_input[{xy,{},{}}], unpack(prev_state)} -- if we're using a grid lstm, hand in a zero vec for the starting memory cell state
        	print('rnn_inputs')
        	print(rnn_inputs)
        	local lst = clones.rnn[xy]:forward(rnn_inputs)
        	print('output')
        	print(lst)
        	rnn_state[xy] = {}

        	--Not Sure here if it will map right or not
        	for i=1,#init_state do table.insert(rnn_state[xy], lst[i]) end -- extract the state, without output
        	predictions[t] = lst[#lst] -- last element is the prediction
        	print('RNN Prediction'..#predictions)
        	--print(predictions[t])
        	--print(#y[t])
        	os.exit()
        	loss = loss + clones.criterion[t]:forward(predictions[xy], y[{xy,{},1}])
       	end
    end
    loss = loss / opt.seq_length

    ------------------ backward pass -------------------
    -- initialize gradient at time t to be zeros (there's no influence from future)
    local drnn_state = {[opt.seq_length] = clone_list(init_state, true)} -- true also zeros the clones
    for x=opt.n_x,1,-1 do
    	for y=opt.n_y,1,-1 do
    		local xy = x*(opt.n_x-1)+y 			-- x,y coordinate in 1D
    		local prev_x = (x-1)*(opt.n_x-1)+y 	-- x-1,y coordinate in 1D
    		local prev_y = x*(opt.n_x-1)+y-1	-- x,y-1 coordinate in 1D

        	-- backprop through loss, and softmax/linear
        	local doutput_t = clones.criterion[t]:backward(predictions[xy], y[{xy,{},1}])
        	table.insert(drnn_state[xy], doutput_t) -- <- drnn_state[t] already has a list of derivative vectors for rnn state pointing to the next time step; just adding the derivative from loss pointing up. 
        	--print(drnn_state[t])
	
        	local dlst = clones.rnn[xy]:backward(rnn_inputs, drnn_state[xy]) -- <- right here, you're appending the doutput_t to the list of dLdh for all layers, then using that big list to backprop into the input and unpacked rnn state vecs at t-1
        	drnn_state[prev_x] = {}
        	drnn_state[prev_y] = {}
        	local skip_index
        	skip_index = 2

        	print(dlst)
        	for k,v in pairs(dlst) do
        	    if k > skip_index then -- k <= skip_index is gradient on inputs, which we dont need
        	        -- note we do k-1 because first item is dembeddings, and then follow the 
        	        -- derivatives of the state, starting at index 2. I know...
        	        drnn_state[t-1][k-skip_index] = v
        	    end
        	end
        end
    end

    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT)
    init_state_global = rnn_state[#rnn_state] -- NOTE: I don't think this needs to be a clone, right?
    -- grad_params:div(opt.seq_length) -- this line should be here but since we use rmsprop it would have no effect. Removing for efficiency
    -- clip gradient element-wise
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)
    return loss, grad_params
end


-- evaluate the loss over an entire split
function eval_split(split_index, max_batches)
    print('evaluating loss over split index ' .. split_index)
    local n = loader.ntest / opt.batch_size
    if max_batches ~= nil then n = math.min(max_batches, n) end

    --loader:reset_batch_pointer(split_index) -- move batch iteration pointer for this split to front
    local loss = 0
    local rnn_state = {[0] = init_state}
    
    for i = 1,n do -- iterate over batches in the split
        -- fetch a batch
        local x, y = loader:next_eval_batch()
        x,y = prepro(x,y)
        -- forward pass
        for t=1,opt.seq_length do
            clones.rnn[t]:evaluate() -- for dropout proper functioning
            local input_mem_cell = torch.zeros(opt.batch_size, opt.rnn_size)
            if opt.gpuid >= 0 then
              input_mem_cell = input_mem_cell:float():cuda()
            end
            rnn_inputs = {input_mem_cell, x[{t,{},{}}], unpack(rnn_state[t-1])} -- if we're using a grid lstm, hand in a zero vec for the starting memory cell state

            local lst = clones.rnn[t]:forward(rnn_inputs)
            rnn_state[t] = {}
            for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end
            prediction = lst[#lst] 
            loss = loss + clones.criterion[t]:forward(prediction, y[{t,{},1}])
        end
        -- carry over lstm state
        rnn_state[0] = rnn_state[#rnn_state]
        print(i .. '/' .. n .. '...')
    end

    loss = loss / opt.seq_length / n
    return loss
end

-- start optimization here
train_losses = {}
val_losses = {}

local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
local iterations = loader.ntrain/opt.batch_size

local iterations_per_epoch = loader.ntrain
local loss0 = nil
for i = 1, iterations  do
    local epoch = opt.batch_size * i / loader.ntrain
    local timer = torch.Timer()

    local _, loss = optim.rmsprop(feval, params, optim_state)
    if opt.accurate_gpu_timing == 1 and opt.gpuid >= 0 then
        --[[
        Note on timing: The reported time can be off because the GPU is invoked async. If one
        wants to have exactly accurate timings one must call cutorch.synchronize() right here.
        I will avoid doing so by default because this can incur computational overhead.
        --]]
        cutorch.synchronize()
    end
    local time = timer:time().real

    local train_loss = loss[1] -- the loss is inside a list, pop it
    train_losses[i] = train_loss

    -- exponential learning rate decay
    if i % loader.ntrain == 0 and opt.learning_rate_decay < 1 then
        if epoch >= opt.learning_rate_decay_after then
            local decay_factor = opt.learning_rate_decay
            optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
            print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
        end
    end
    
    -- every now and then or on last iteration
    if i % opt.eval_val_every == 0 or i == iterations then
    	    -- evaluate loss on validation data
    	    local val_loss = eval_split(2) -- 2 = validation
    	    val_losses[i] = val_loss
	
    	    local savefile = string.format('%s/lm_%s_epoch%.2f_%.4f.t7', opt.checkpoint_dir, opt.savefile, epoch, val_loss)
    	    print('saving checkpoint to ' .. savefile)
    	    local checkpoint = {}
    	    checkpoint.protos = protos
    	    checkpoint.opt = opt
    	    checkpoint.train_losses = train_losses
    	    checkpoint.val_loss = val_loss
    	    checkpoint.val_losses = val_losses
    	    checkpoint.i = i
    	    checkpoint.epoch = epoch
    	    checkpoint.vocab = loader.vocab_mapping
    	    torch.save(savefile, checkpoint)
    end


    if i % opt.print_every == 0 then
        print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.4fs", i, iterations, epoch, train_loss, grad_params:norm() / params:norm(), time))
    end
   
    if i % 10 == 0 then collectgarbage() end

    -- handle early stopping if things are going really bad
    if loss[1] ~= loss[1] then
        print('loss is NaN.  This usually indicates a bug.  Please check the issues page for existing issues, or create a new issue, if none exist.  Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?')
        break -- halt
    end
    if loss0 == nil then loss0 = loss[1] end
    if loss[1] > loss0 * 3 then
        print('loss is exploding, aborting.')
        break -- halt
    end
end