-- Based on https://github.com/Element-Research/dpnn/blob/master/ReverseTable.lua
-- By Nicholas Leonard

-- The module takes an input of 2D table represented as 1D sequence of elements and reverses it
-- from right top corner for implementing the layers of multidirectional RNN


-- Takes an input {1,2,3,4,5,6,7,8,9,0,1,2} and outputs {4,3,2,1,8,7,6,5,2,1,0,9}
-- like  1 2 3 4*  4 3 2 1
--       5 6 7 8 ->8 7 6 5    
--       9 0 1 2   2 1 0 1   
require 'nn'
local SymetricTable, parent = torch.class("nn.SymmetricTable", "nn.Module")

function SymetricTable:__init(width, height)
   parent.__init(self)
   self.width = width
   self.height = height

   self.output = {}
   self.gradInput = {}
end

function SymetricTable:updateOutput(inputTable)
   assert(torch.type(inputTable) == 'table', "Expecting table at arg 1")

   -- empty output table
   for k,v in ipairs(self.output) do
      self.output[k] = nil
   end
   
   -- reverse input
   local k = 1
   for y = 1,self.height do 
      for x= self.width, 1,-1 do
         self.output[k] = inputTable[(y-1)*self.width+x]
         k = k + 1
      end
   end

   return self.output
end

function SymetricTable:updateGradInput(inputTable, gradOutputTable)
   
   -- empty gradInput table
   for k,v in ipairs(self.gradInput) do
      self.gradInput[k] = nil
   end
   
   -- reverse gradOutput
   local k = 1
   for y = 1,self.height do 
      for x= self.width, 1,-1 do
         self.gradInput[k] = gradOutputTable[(y-1)*self.width+x]
         k = k + 1
      end
   end
   return self.gradInput
end