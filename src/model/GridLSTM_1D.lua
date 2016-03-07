------------------------------------------------------------------------
--[[ Grid_LSTM ]]--
-- Implementation of GridLSTM based on RNN package for speed and efficiency
-- Ref. A.: http://arxiv.org/abs/1507.01526 (Nal et al.)
-- Based on Corey Lynch Implementation of 2D GRID (give link)

-- Processes the sequence one timestep (forward/backward) at a time. 
-- A call to backward only keeps a log of the gradOutputs and scales.
-- Back-Propagation Through Time (BPTT) is done when updateParameters
-- is called. The Module keeps a list of all previous representations 
-- (Module.outputs), including intermediate ones for BPTT.
-- To use this module with batches, we suggest using different 
-- sequences of the same size within a batch and calling 
-- updateParameters() at the end of the Sequence. 
-- Note that this won't work with modules that use more than the
-- output attribute to keep track of their internal state between 
-- forward and backward.
------------------------------------------------------------------------


------------------------------------------------------------------------
--[[ To Do ]]--
-- Grad2D ouptut saving
--[1] Add FastLSTM
--[2] Right Padding at the edges
require 'nn'
require 'nngraph'
--nngraph.setDebug(true)
assert(not nn.GridLSTM_1D, "update nnx package : luarocks install nnx")
local GridLSTM_1D, parent = torch.class('nn.GridLSTM_1D', 'nn.AbstractRecurrent')

function GridLSTM_1D:__init(input_unit_size, rnn_size, n_layers, rho, should_tie_weights, dropout )
  self.layer = layer
  parent.__init(self, rho)

  self.input_size_t = 0
  self.n_layers = n_layers
  self.dropout = dropout
  self.input_unit_size = input_unit_size
  self.rnn_size = rnn_size
  self.should_tie_weights = should_tie_weights
  
  -- Build model here
  self.recurrentModule = self:buildModel()

  -- make it work with nn.Container
  self.modules[1] = self.recurrentModule
  self.sharedClones[1] = self.recurrentModule

  -- for output(0), cell(0) and gradCell(T)
  self.zeroTensor = torch.Tensor() 

  self.rnn_states = {}
  self.drnn_states = {} -- true also zeros the clones

end

-------------------------- factory methods -----------------------------

--[[
  This is called once per dimension inside a grid LSTM block to create the gated
  update of the dimension's hidden state and memory cell.

  It takes h_t and h_d, the hidden states from the temporal and 
  depth dimensions respectively, as well as prev_c, the 
  dimension's previous memory cell.

  It returns next_c, next_h along the dimension, using a standard
  lstm gated update, conditioned on the concatenated time and 
  depth hidden states.
--]]

-- This should be later modified to use nn.FastLSTM
local function lstm(h_t, h_d, prev_c, rnn_size)
  local all_input_sums = nn.CAddTable()({h_t, h_d})
  local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
  local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
  -- decode the gates
  local in_gate = nn.Sigmoid()(n1)
  local forget_gate = nn.Sigmoid()(n2)
  local out_gate = nn.Sigmoid()(n3)
  -- decode the write inputs
  local in_transform = nn.Tanh()(n4)
  -- perform the LSTM update
  --print(forget_gate)
  --print(prev_c)
  local next_c           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_c}),
      nn.CMulTable()({in_gate,     in_transform})
    })
  -- gated cells form the output
  local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
  return next_c, next_h
end


--[[
  GridLSTM:
    1) Map input x into memory and hidden cells m(1), h(1) along the depth dimension.
    2) Concatenate previous hidden states from time and depth dimensions, [h(1), h(2)] into H.
    3) Forward the time LSTM, LSTM_2(H) -> h(2)', m(2)'.
    4) Concatenate transformed h(2)' and h(1) into H' = [h(1), h(2)']
    5) Forward the depth LSTM, LSTM_1(H') -> h(1)', m(1)'
    6) Either repeat 2-5 for another layer or map h(1)', the final hidden state along the depth 
       dimension, to a character prediction.
  --]]

function GridLSTM_1D:buildModel()

  -- There will be 2*n+1 inputs
	local inputs = {}
	-- init input dimensions {input c, input h, x-dim c, x-dim h, y-dim c, y-dim h}
	table.insert(inputs, nn.Identity()()) -- input c for depth dimension
  table.insert(inputs, nn.Identity()()) -- input h for depth dimension
  for L = 1,self.n_layers do
    table.insert(inputs, nn.Identity()()) -- prev_c[L] for t dimension
    table.insert(inputs, nn.Identity()()) -- prev_h[L] for t dimension
  end

  -- Sharing might not work with current setup
	local shared_weights
  if self.should_tie_weights == 1 then 
    shared_weights = {nn.Linear(self.rnn_size, 4 * self.rnn_size),
                      nn.Linear(self.rnn_size, 4 * self.rnn_size)} 
  end

  local outputs_t = {} -- Outputs being handed to the next time step along the y dimension
  local outputs_d = {} -- Outputs being handed from one layer to the next along the depth dimension
  for L = 1,self.n_layers do

    local prev_c_t = inputs[L*2+1]
    local prev_h_t = inputs[L*2+2]

    if L == 1 then
      -- We're in the first layer
      prev_c_d = nn.Linear(self.input_unit_size, self.rnn_size)(inputs[1]) -- input_c_d: the starting depth dimension memory cell, just a zero vec.
      prev_h_d = nn.Linear(self.input_unit_size, self.rnn_size)(inputs[2]) -- input_h_d: the starting depth dimension hidden state. We map a char into hidden space via a lookup table
    else 
      -- We're in the higher layers 2...N
      -- Take hidden and memory cell from layers below
      prev_c_d = outputs_d[((L-1)*2)-1]
      prev_h_d = outputs_d[((L-1)*2)]
      if self.dropout > 0 then prev_h_d = nn.Dropout(self.dropout)(prev_h_d):annotate{name='drop_' .. L} end -- apply dropout, if any
    end

       -- Evaluate the input sums at once for efficiency
    local t2h_t = nn.Linear(self.rnn_size, 4 * self.rnn_size)(prev_h_t):annotate{name='t2h'}
    local d2h_t = nn.Linear(self.rnn_size, 4 * self.rnn_size)(prev_h_d):annotate{name='h2h'}
    	
    -- Get transformed memory and hidden states pointing in the time direction first
    local next_c_t, next_h_t = lstm(t2h_t, d2h_t, prev_c_t, self.rnn_size)

     -- Pass memory cell and hidden state to next timestep
    table.insert(outputs_t, next_c_t)
    table.insert(outputs_t, next_h_t)

    -- Evaluate the input sums at once for efficiency
    local t2h_d = nn.Linear(self.rnn_size, 4 * self.rnn_size)(next_h_t):annotate{name='i2h'}:annotate{name='eval_1'}
    local d2h_d = nn.Linear(self.rnn_size, 4 * self.rnn_size)(prev_h_d):annotate{name='h2h'}:annotate{name='eval_2'}
	 
    -- See section 3.5, "Weight Sharing" of http://arxiv.org/pdf/1507.01526.pdf
    -- The weights along the temporal dimension are already tied (cloned many times in train.lua)
    -- Here we can tie the weights along the depth dimension. Having invariance in computation
    -- along the depth appears to be critical to solving the 15 digit addition problem w/ high accy.
    -- See fig 4. to compare tied vs untied grid lstms on this task.
    if should_tie_weights == 1 then
      print("tying weights along the depth dimension")
      t2h_d.data.module:share(shared_weights[1], 'weight', 'bias', 'gradWeight', 'gradBias')
      d2h_d.data.module:share(shared_weights[2], 'weight', 'bias', 'gradWeight', 'gradBias')
    end
    
    -- Create the lstm gated update pointing in the depth direction.
    -- We 'prioritize' the depth dimension by using the updated temporal hidden state as input
    -- instead of the previous temporal hidden state. This implements Section 3.2, "Priority Dimensions"
    local next_c_d, next_h_d = lstm(t2h_d, d2h_d, prev_c_d, self.rnn_size)
  
    -- Pass the depth dimension memory cell and hidden state to layer above
    table.insert(outputs_d, next_c_d)
    table.insert(outputs_d, next_h_d)
  end

  local top_h = outputs_d[#outputs_d]
  if self.dropout > 0 then top_h = nn.Dropout(self.dropout)(top_h) end
  local proj = nn.Linear(rnn_size, self.input_unit_size)(top_h):annotate{name='decoder'}
  table.insert(outputs_t, proj)

  return nn.gModule(inputs, outputs_t)
end



------------------------- forward backward -----------------------------
function GridLSTM_1D:updateOutput(input)

  if(self.layer == 2) then print(input) end

  if input[1]:dim() == 2 then
      self.zeroTensor:resize(input[1]:size(1), self.rnn_size):zero()
  else
      self.zeroTensor:resize(self.rnn_size):zero()
  end

	-- Concatinate states of previous dimensions x and y into prev_state along each layer
  prev_state = {}
  for L=1,self.n_layers do
    local prev_h_t, prev_c_t 
    if self.step == 1 then
      prev_c_t = self.zeroTensor
      prev_h_t = self.zeroTensor
    else
      -- previous output and cell of this module given Layer
      prev_c_t = self.rnn_states[self.step-1][L]
      prev_h_t = self.rnn_states[self.step-1][L+1]
    end

    table.insert(prev_state, prev_c_t)  -- prev x_c 
    table.insert(prev_state, prev_h_t)  -- prev x_h
  end

	-- Feed forward 
	rnn_inputs = {input[1],input[2], unpack(prev_state)} 

	local output
	if self.train ~= false then
    self:recycle()
    local recurrentModule = self:getStepModule(self.step)
    -- the actual forward propagation
    output = recurrentModule:updateOutput{unpack(rnn_inputs)}
	else
    output = self.recurrentModule:updateOutput{unpack(rnn_inputs)}
	end

  -- Extract input information
  self.outputs[self.step] = output[2*self.n_layers+1]
  self.rnn_states[self.step] = {}
  for i=1,(2*self.n_layers) do table.insert(self.rnn_states[self.step], output[i]) end -- extract the state, without output

  self.output = output[2*self.n_layers+1]
  self.step = self.step + 1

  self.gradPrevOutput = nil
  self.updateGradInputStep = nil
  self.accGradParametersStep = nil

	return self.output -- return h
end



function GridLSTM_1D:_updateGradInput(input, gradOutput)
  if(self.layer == 1) then print(gradOutput) end

	assert(self.step > 1, "expecting at least one updateOutput")
	local step = self.updateGradInputStep - 1
  assert(step >= 1)

  -- set the output/gradOutput states of current Module
  local recurrentModule = self:getStepModule(step)

  -- backward propagate through this step
  if self.gradPrevOutput then
    self._gradOutputs[step] = nn.rnn.recursiveCopy(self._gradOutputs[step], self.gradPrevOutput)
  --  nn.rnn.recursiveAdd(self._gradOutputs[step], gradOutput)
  --  gradOutput =  self._gradOutputs[step]
  end
  -- Construct input

  local rnn_state = {}
  if self.rnn_states[step-1] then
    for L=1, self.n_layers do
        -- previous output and cell of this module given Layer
        table.insert(rnn_state, self.rnn_states[step-1][L])          -- prev x_c 
        table.insert(rnn_state, self.rnn_states[step-1][L+1])        -- prev x_h
    end
  else
    for i=1, (2*self.n_layers) do table.insert(rnn_state, self.zeroTensor) end  
  end

  local drnn_state = {}
  if self.drnn_states[step] then
    for L=1, self.n_layers do
        -- previous output and cell of this module given Layer
        table.insert(drnn_state, self.drnn_states[step][L])          -- prev x_c 
        table.insert(drnn_state, self.drnn_states[step][L+1])        -- prev x_h
    end
  else
    for i=1,(2*self.n_layers) do table.insert(drnn_state, self.zeroTensor) end
  end

  local inputTable = {input[1], input[2], unpack(rnn_state)}
  local outputTable = {unpack(drnn_state)}
  table.insert(outputTable, gradOutput)

  local gradInputTable = recurrentModule:updateGradInput(inputTable, outputTable)
  
  local gradInput = gradInputTable[2]
  self.gradPrevOutput = gradOutput -- still need to correctly think
  --for i = 1,(self.n_layers) do table.insert(self.gradPrevOutput, gradInputTable[2*i+1]) end
  
  self.drnn_states[step-1] = {}
  for i = 3,(2*self.n_layers)+2 do table.insert(self.drnn_states[step-1], gradInputTable[i]) end
	return gradInput
end


function GridLSTM_1D:_accGradParameters(input, gradOutput, scale)
  
  local step = self.accGradParametersStep - 1

  assert(step >= 1)

  -- set the output/gradOutput states of current Module
  local recurrentModule = self:getStepModule(step)
   
  -- Construct input
  --local input_mem_cell = self.zeroTensor
  local rnn_state = {}
  if self.rnn_states[step-1] then
    for L=1, self.n_layers do
        -- previous output and cell of this module given Layer
        table.insert(rnn_state, self.rnn_states[step-1][L])          -- prev x_c 
        table.insert(rnn_state, self.rnn_states[step-1][L+1])        -- prev x_h
    end
  else
    for i=1, (2*self.n_layers) do table.insert(rnn_state, self.zeroTensor) end  
  end

  local drnn_state = {}
  if self.drnn_states[step] then
    for L=1, self.n_layers do
        -- previous output and cell of this module given Layer
        table.insert(drnn_state, self.drnn_states[step][L])          -- prev x_c 
        table.insert(drnn_state, self.drnn_states[step][L+1])        -- prev x_h
    end
  else
    for i=1,(2*self.n_layers) do table.insert(drnn_state, self.zeroTensor) end
  end

  local gradOutput = (step == self.step - 1) and gradOutput or self._gradOutputs[step] 
  table.insert(drnn_state, gradOutput)

  local inputTable = {input[1],input[2], unpack(rnn_state)}
  local gradOutputTable = drnn_state

  recurrentModule:accGradParameters(inputTable, gradOutputTable, scale)
  
  return gradInput
end
