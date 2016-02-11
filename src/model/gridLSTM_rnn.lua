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
assert(not nn.GridLSTM, "update nnx package : luarocks install nnx")
local GridLSTM, parent = torch.class('nn.GridLSTM', 'nn.AbstractRecurrent')

function GridLSTM:__init(input_unit_size, input_size_x, input_size_y, rnn_size, rho, should_tie_weights, layer)
  self.layer = layer
  parent.__init(self, rho)

  self.input_size_x = input_size_x
  self.input_size_y = input_size_y
  self.input_size = input_size_x * input_size_y
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
  self.drnn_states = {}--{[opt.seq_length] = clone_list(init_state, true)} -- true also zeros the clones
  for i = -self.input_size_x, self.input_size do -- padding for states
  		self.drnn_states[i] = {}
	  for j=1,4 do table.insert(self.drnn_states[i], self.zeroTensor) end
  end

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
local function lstm(h_x, h_y, h_d, prev_c, rnn_size)
  local all_input_sums = nn.CAddTable()({h_x, h_y, h_d})
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

function GridLSTM:buildModel()

  -- There will be 2*n+1 inputs
	local inputs = {}

	-- init input dimensions {input c, input h, x-dim c, x-dim h, y-dim c, y-dim h}
	for L = 1,6 do table.insert(inputs, nn.Identity()()) end

  -- Sharing might not work with current setup
	local shared_weights
  if self.should_tie_weights == 1 then 
    shared_weights = {nn.Linear(self.rnn_size, 4 * self.rnn_size),
                      nn.Linear(self.rnn_size, 4 * self.rnn_size), 
                      nn.Linear(self.rnn_size, 4 * self.rnn_size)} 
  end

  local outputs_t = {} -- Outputs being handed to the next time step along the y dimension
  local outputs_d = {} -- Outputs being handed from one layer to the next along the depth dimension

  -- We're in the first layer       
  prev_c_d = nn.Linear(self.input_unit_size, self.rnn_size)(inputs[1]):annotate{name='input C'} -- input_c_d 
  prev_h_d = nn.Linear(self.input_unit_size, self.rnn_size)(inputs[2]):annotate{name='input H'} -- input_h_d

  local prev_c_x = inputs[3]
  local prev_h_x = inputs[4]
	
  local prev_c_y = inputs[5]
  local prev_h_y = inputs[6]

  -- Evaluate the input sums at once for efficiency
  local x2h_t = nn.Linear(self.rnn_size, 4 * self.rnn_size)(prev_h_x):annotate{name='x2h'}
  local y2h_t = nn.Linear(self.rnn_size, 4 * self.rnn_size)(prev_h_y):annotate{name='y2h'}
  local d2h_t = nn.Linear(self.rnn_size, 4 * self.rnn_size)(prev_h_d):annotate{name='h2h'}
  	
  -- Get transformed memory and hidden states pointing in the time direction first
  local next_c_x, next_h_x = lstm(x2h_t, y2h_t, d2h_t, prev_c_x, self.rnn_size)
  local next_c_y, next_h_y = lstm(x2h_t, y2h_t, d2h_t, prev_c_y, self.rnn_size)
  --local next_c_d, next_h_d = lstm(x2h_t, y2h_t, d2h_t, prev_c_d, self.rnn_size)

  -- Evaluate the input sums at once for efficiency
  local x2h_d = nn.Linear(self.rnn_size, 4 * self.rnn_size)(next_h_x):annotate{name='i2h'}:annotate{name='eval_1'}
  local y2h_d = nn.Linear(self.rnn_size, 4 * self.rnn_size)(next_h_y):annotate{name='i2h'}:annotate{name='eval_2'}
  local d2h_d = nn.Linear(self.rnn_size, 4 * self.rnn_size)(prev_h_d):annotate{name='h2h'}:annotate{name='eval_2'}
	
  -- See section 3.5, "Weight Sharing" of http://arxiv.org/pdf/1507.01526.pdf
  -- The weights along the temporal dimension are already tied (cloned many times in train.lua)
  -- Here we can tie the weights along the depth dimension. Having invariance in computation
  -- along the depth appears to be critical to solving the 15 digit addition problem w/ high accy.
  -- See fig 4. to compare tied vs untied grid lstms on this task.
  if should_tie_weights == 1 then
    print("tying weights along the depth dimension")
    x2h_d.data.module:share(shared_weights[1], 'weight', 'bias', 'gradWeight', 'gradBias')
    y2h_d.data.module:share(shared_weights[2], 'weight', 'bias', 'gradWeight', 'gradBias')
    d2h_d.data.module:share(shared_weights[3], 'weight', 'bias', 'gradWeight', 'gradBias')
  end
   	
  -- Create the lstm gated update pointing in the depth direction.
  -- We 'prioritize' the depth dimension by using the updated temporal hidden state as input
  -- instead of the previous temporal hidden state. This implements Section 3.2, "Priority Dimensions"
  local next_c_d, next_h_d = lstm(x2h_d, y2h_d, d2h_d, prev_c_d, self.rnn_size)

  -- Pass the depth dimension memory cell and hidden state to layer above
	table.insert(outputs_t, next_c_d)
  table.insert(outputs_t, next_h_d)

  -- Pass memory cell and hidden state to next timestep
  table.insert(outputs_t, next_c_x)
  table.insert(outputs_t, next_h_x)
  table.insert(outputs_t, next_c_y)
  table.insert(outputs_t, next_h_y)

  return nn.gModule(inputs, outputs_t)
end


------------------------- forward backward -----------------------------
local function getPreviousState(step, state, zeroTensor, input_size_x, backward)
  local c = backward or 1
  prev_state = {}
  local prev_h_x, prev_c_x, prev_h_y, prev_c_y

  -- No input from edges of x
  if step < 1 or step%input_size_x == 1 then
    prev_h_x = zeroTensor
    prev_c_x = zeroTensor
  else
    prev_c_x = state[step-c][1]
    prev_h_x = state[step-c][2]
  end

  -- No input from upper edge
  if step < input_size_x+1 then
    prev_h_y = zeroTensor
    prev_c_y = zeroTensor
  else
    -- previous output and cell of this module given Layer
    prev_c_y = state[step-c*input_size_x][3]  
    prev_h_y = state[step-c*input_size_x][4]
  end

  table.insert(prev_state, prev_c_x)  -- prev x_c 
  table.insert(prev_state, prev_h_x)  -- prev x_h
  
  table.insert(prev_state, prev_c_y)  -- prev y_c 
  table.insert(prev_state, prev_h_y)  -- prev y_h 

  return prev_state
end

function GridLSTM:updateOutput(input)
  if(self.layer == 2) then print(input) end

  if input[1]:dim() == 2 then
      self.zeroTensor:resize(input[1]:size(1), self.rnn_size):zero()
  else
      self.zeroTensor:resize(self.rnn_size):zero()
  end

	-- Concatinate states of previous dimensions x and y into prev_state along each layer
  local prev_state = getPreviousState(self.step, self.rnn_states, self.zeroTensor, self.input_size_x,1)

	-- Feed forward 
  -- Input[1] for input to Cell dim 
  -- Input[2] for input to Depth dim
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
  self.outputs[self.step] = {}
  self.rnn_states[self.step] = {}
  for i=3,6 do table.insert(self.rnn_states[self.step], output[i]) end -- extract the state, without output
  
  self.output = {output[1], output[2]} 
  self.outputs[self.step] = self.output
  self.step = self.step + 1

  self.gradPrevOutput = nil
  self.updateGradInputStep = nil
  self.accGradParametersStep = nil

	return self.output -- return {c, h} 
end


-- Also discuss Edge cases when there is only one available previous step
-- Warning hasn't that discussed please
function GridLSTM:_updateGradInput(input, gradOutput)
  if(self.layer == 1) then print(gradOutput) end

	assert(self.step > 1, "expecting at least one updateOutput")
	local step = self.updateGradInputStep - 1
  assert(step >= 1)
  -- set the output/gradOutput states of current Module
  local recurrentModule = self:getStepModule(step)

  -- backward propagate through this step
  if self.gradPrevOutput then
    self._gradOutputs[step] = nn.rnn.recursiveCopy(self._gradOutputs[step], self.gradPrevOutput)
    nn.rnn.recursiveAdd(self._gradOutputs[step], gradOutput)
    gradOutput = self._gradOutputs[step]
  end

  -- Construct input
  local rnn_state = getPreviousState(step, self.rnn_states, self.zeroTensor, self.input_size_x, 1)
  local drnn_state = getPreviousState(step, self.drnn_states, self.zeroTensor, self.input_size_x, 0)

  local inputTable = {input[1], input[2], unpack(rnn_state)}
  local outputTable = {gradOutput[1], gradOutput[2], unpack(self.drnn_states[step])}

  local gradInputTable = recurrentModule:updateGradInput(inputTable, outputTable)

  local gradInput = {gradInputTable[1], gradInputTable[2]}
  self.gradPrevOutput = gradOutput

  self.drnn_states[step-1][1] = gradInputTable[3]	  -- prev x_c 
  self.drnn_states[step-1][2] = gradInputTable[4]	  -- prev x_h 

  self.drnn_states[step-self.input_size_x][3] = gradInputTable[5]	-- prev y_c 
  self.drnn_states[step-self.input_size_x][4] = gradInputTable[6]	-- prev y_h 

	return gradInput
end


function GridLSTM:_accGradParameters(input, gradOutput, scale)
  
  local step = self.accGradParametersStep - 1

  assert(step >= 1)

  -- set the output/gradOutput states of current Module
  local recurrentModule = self:getStepModule(step)
   
  -- Construct input
  --local input_mem_cell = self.zeroTensor
  local rnn_state = getPreviousState(step, self.rnn_states, self.zeroTensor, self.input_size_x, 1)
  local drnn_state = getPreviousState(step, self.drnn_states, self.zeroTensor, self.input_size_x, 0)

  local inputTable = {input[1],input[2], unpack(rnn_state)}
  local gradOutput = (step > self.step - self.input_size_x) and gradOutput or self._gradOutputs[step] 
  -- (-self.input_size_x)

  local gradOutputTable = {gradOutput[1], gradOutput[2], unpack(self.drnn_states[step])}

  recurrentModule:accGradParameters(inputTable, gradOutputTable, scale)
  
  return gradInput
end
