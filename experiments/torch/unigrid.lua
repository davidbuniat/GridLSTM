local lstm = require 'lstm'
local unigrid = {}
function unigrid.unigrid(size, n, prev_h, prev_m)
    local h = prev_h
    local m = prev_m
    for i=1,n do
        h, m = lstm.lstm(size, h, m)
    end
    return h, m
end
return unigrid