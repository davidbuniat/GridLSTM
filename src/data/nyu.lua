require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nnx'      -- provides a normalization operator
require 'nn'
local matio = require 'matio'
local data = {}
data.__index = data

print(sys.COLORS.red ..  '==> loading dataset')
images = matio.load('nyu_dataset.mat', 'rgbs');
labels = matio.load('nyu_dataset.mat', 'labels');

print '==> preprocessing data'

local height = (#images)[2]
local width = (#images)[3]
	
images = images:permute(1,4,2,3)
images = images/256
-- shuffle dataset: get shuffled indices in this variable:
local labelsShuffle = torch.randperm((#labels)[1])

local portionTrain = 0.7 -- 55% is train data
local portionTest = 0.25 -- 33% is valid data, rest is test data

local trsize = torch.floor(labelsShuffle:size(1)*portionTrain)
local tesize = torch.floor(labelsShuffle:size(1)*portionTest)
local valsize = labelsShuffle:size(1) - trsize - tesize

-- create train set:
trainData = {
   x = torch.Tensor(trsize, 3, height, width),
   y = torch.Tensor(trsize, 1, height, width ),
   size = trsize,
   pos = 0
}

--create validation set:
testData = {
    x = torch.Tensor(tesize, 3, height, width ),
    y = torch.Tensor(tesize, 1,height, width ),
    size = tesize,
    pos = 0
}

--create test set:
valData = {
    x = torch.Tensor(valsize, 3, height, width),
    y = torch.Tensor(valsize, 1, height, width),
    size = valsize,
    pos = 0
}

for i=1,trsize do
   trainData.x[i] = images[labelsShuffle[i]]:clone()
   trainData.y[i] = labels[labelsShuffle[i]]:clone()
end

for i=trsize+1,(trsize+tesize) do
   	testData.x[i-trsize] = images[labelsShuffle[i]]:clone()
   	testData.y[i-trsize] = labels[labelsShuffle[i]]:clone()
end

for i=trsize+tesize+1,valsize+tesize+trsize do
   	valData.x[i-trsize-tesize] = images[labelsShuffle[i]]:clone()
   	valData.y[i-trsize-tesize] = labels[labelsShuffle[i]]:clone()
end

-- remove from memory temp image files:
images = nil
labels = nil
collectgarbage();
	

-- making float
trainData.x = trainData.x:float()
valData.x = valData.x:float()
testData.x = testData.x:float()

trainData.y = trainData.y:float()
valData.y = valData.y:float()
testData.y = testData.y:float()

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> visualizing data (Skipping)')

-- Visualization is quite easy, using image.display(). Check out:
-- help(image.display), for more info about options.

if false then
   local first256Samples_y = {trainData.x[{ {1},1 }],trainData.y[{ {1},1 }]}
   --image.display{image=first256Samples_y, nrow=16, legend='Some training examples: Y channel'}
   local first256Samples_y = {valData.x[{ {1},1 }],valData.y[{ {1},1 }]}
   --image.display{image=first256Samples_y, nrow=16, legend='Some valid examples: Y channel'}
   local first256Samples_y = {testData.x[{ {1},1 }],testData.y[{ {1},1 }]}
   --image.display{image=first256Samples_y, nrow=16, legend='Some testing examples: Y channel'}
end

print '==> Data is ready'

data.traindataset = trainData
data.testdataset = testData
data.validationdataset = valData


return data