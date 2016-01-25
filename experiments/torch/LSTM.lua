local lstm = {}
function lstm.lstm(size, prev_h, prev_m)
    local gates = nn.Linear(size, 4*size)(prev_h)
    

    local reshaped = nn.Reshape(4,size)(gates)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    local input_gate = nn.Sigmoid()(n1)
    local forget_gate = nn.Sigmoid()(n2)
    local output_gate = nn.Sigmoid()(n3)
    local memory_transform = nn.Tanh()(n4)
    local next_m = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_m}),
        nn.CMulTable()({input_gate, memory_transform})
      })
    local next_h = nn.CMulTable()({output_gate, nn.Tanh()(next_m)})
    return next_h, next_m
end
return lstm


function lstm(h_t, h_d, prev_c, rnn_size)
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
  local next_c           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_c}),
      nn.CMulTable()({in_gate,     in_transform})
    })
  -- gated cells form the output
  local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
  return next_c, next_h
end