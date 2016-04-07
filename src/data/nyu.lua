require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nnx'      -- provides a normalization operator
require 'nn'
 matio = require 'matio'
local data = {}
data.__index = data

print(sys.COLORS.red ..  '==> loading dataset')
images = matio.load('data/nyu_dataset.mat', 'rgbs');
labels = matio.load('data/nyu_dataset.mat', 'depths');

print '==> preprocessing data'

local height = (#images)[2]
local width = (#images)[3]
	
images = images:permute(1,4,2,3)
images = images/256


-- shuffle dataset: get shuffled indices in this variable:
--local labelsShuffle = torch.randperm((#labels)[1])
local n_labels = (#labels)[1]

local portionTrain = 0.55 -- 55% is train data
local portionTest = 0.4 -- 33% is valid data, rest is test data

local trsize = torch.floor(n_labels*portionTrain)
local tesize = torch.floor(n_labels*portionTest)
local valsize = (n_labels) - trsize - tesize

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
   trainData.x[i] = images[i]:clone()
   trainData.y[i] = labels[i]:clone()
end

for i=trsize+1,(trsize+tesize) do
   	testData.x[i-trsize] = images[i]:clone()
   	testData.y[i-trsize] = labels[i]:clone()
end

for i=trsize+tesize+1,valsize+tesize+trsize do
   	valData.x[i-trsize-tesize] = images[i]:clone()
   	valData.y[i-trsize-tesize] = labels[i]:clone()
end

-- remove from memory temp image files:
images = nil
labels = nil
collectgarbage();
	



-- Preprocessing requires a floating point representation (the original
  -- data is stored on bytes). Types can be easily converted in Torch, 
  -- in general by doing: dst = src:type('torch.TypeTensor'), 
  -- where Type=='Float','Double','Byte','Int',... Shortcuts are provided
  -- for simplicity (float(),double(),cuda(),...):

trainData.x = trainData.x:float()
valData.x = valData.x:float()
testData.x = testData.x:float()

trainData.y = trainData.y:float()
valData.y = valData.y:float()
testData.y = testData.y:float()

-- We now preprocess the data. Preprocessing is crucial
-- when applying pretty much any kind of machine learning algorithm.

-- For natural images, we use several intuitive tricks:
--   + images are mapped into YUV space, to separate luminance information
--     from color information
--   + the luminance channel (Y) is locally normalized, using a contrastive
--     normalization operator: for each neighborhood, defined by a Gaussian
--     kernel, the mean is suppressed, and the standard deviation is normalized
--     to one.
--   + color channels are normalized globally, across the entire dataset;
--     as a result, each color component has 0-mean and 1-norm across the dataset.

-- Convert all images to YUV
 print '==> preprocessing data: colorspace RGB -> YUV'

 for i = 1,trainData.size do 
    trainData.x[i] = image.rgb2yuv(trainData.x[i])
    --trainData.labels[i] = image.rgb2yuv(trainData.labels[i])
 end

 for i = 1,valData.size do
    valData.x[i] = image.rgb2yuv(valData.x[i])
    --valData.labels[i] = image.rgb2yuv(valData.labels[i])
 end

 for i = 1,testData.size do
    testData.x[i] = image.rgb2yuv(testData.x[i])
    --testData.labels[i] = image.rgb2yuv(testData.labels[i])
 end

   
-- Name channels for convenience
local channels = {'y', 'u', 'v'}

-- Normalize each channel, and store mean/std
-- per channel. These values are important, as they are part of
-- the trainable parameters. At test time, test data will be normalized
-- using these values.
print(sys.COLORS.red ..  '==> preprocessing data: normalize each feature (channel) globally')
local mean = {}
local std = {}

local l_mean;
local l_std;

for i,channel in ipairs(channels) do
   -- normalize each channel globally:
   mean[i] = trainData.x[{ {},i,{},{} }]:mean()
   std[i] = trainData.x[{ {},i,{},{} }]:std()
   trainData.x[{ {},i,{},{} }]:add(-mean[i])
   trainData.x[{ {},i,{},{} }]:div(std[i])
end


-- Normalize test data, using the training means/stds
for i,channel in ipairs(channels) do
  -- normalize each channel globally:
  valData.x[{ {},i,{},{} }]:add(-mean[i])
  valData.x[{ {},i,{},{} }]:div(std[i])
end



-- Normalize test data, using the training means/stds
for i,channel in ipairs(channels) do
   -- normalize each channel globally:
   testData.x[{ {},i,{},{} }]:add(-mean[i])
   testData.x[{ {},i,{},{} }]:div(std[i])
end


-- Normalize label data
l_mean = trainData.y:mean()
l_std = trainData.y:std()
trainData.y:add(-l_mean)
trainData.y:div(l_std)

valData.y:add(-l_mean)
valData.y:div(l_std)

testData.y:add(-l_mean)
testData.y:div(l_std)

-- Local contrast normalization is needed in the face dataset as the dataset is already in this form:
print(sys.COLORS.red ..  '==> preprocessing data: normalize all three channels locally')

-- Define the normalization neighborhood:
local neighborhood = image.gaussian1D(5) -- 5 for face detector training

-- Define our local normalization operator (It is an actual nn module, 
-- which could be inserted into a trainable model):
local normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()

-- Normalize all channels locally:
for c in ipairs(channels) do
   for i = 1,trainData.size do
      trainData.x[{ i,{c},{},{} }] = normalization:forward(trainData.x[{ i,{c},{},{} }])
   end
   for i = 1,valData.size do
      valData.x[{ i,{c},{},{} }] = normalization:forward(valData.x[{ i,{c},{},{} }])
   end
   for i = 1,testData.size do
      testData.x[{ i,{c},{},{} }] = normalization:forward(testData.x[{ i,{c},{},{} }])
   end
end

--for i = 1,trainData.size do
--   trainData.y[{ i,{1},{},{} }] = normalization:forward(trainData.y[{ i,{1},{},{} }])
--end
--for i = 1,valData.size do
--   valData.y[{ i,{1},{},{} }] = normalization:forward(valData.y[{ i,{1},{},{} }])
--end
--for i = 1,testData.size do
--   testData.y[{ i,{1},{},{} }] = normalization:forward(testData.y[{ i,{1},{},{} }])
--end

 ----------------------------------------------------------------------
 print(sys.COLORS.red ..  '==> verify statistics')
 
 -- It's always good practice to verify that data is properly
 -- normalized.
 
 for i,channel in ipairs(channels) do
    local trainMean = trainData.x[{ {},i }]:mean()
    local trainStd = trainData.x[{ {},i }]:std()
 
    local valMean = valData.x[{ {},i }]:mean()
    local valStd = valData.x[{ {},i }]:std()
 
    local testMean = testData.x[{ {},i }]:mean()
    local testStd = testData.x[{ {},i }]:std()
 
    print('training data, '..channel..'-channel, mean: ' .. trainMean)
    print('training data, '..channel..'-channel, standard deviation: ' .. trainStd)
 
    print('test data, '..channel..'-channel, mean: ' .. valMean)
    print('test data, '..channel..'-channel, standard deviation: ' .. valStd)
 
    print('test data, '..channel..'-channel, mean: ' .. testMean)
    print('test data, '..channel..'-channel, standard deviation: ' .. testStd)
 
 end

print('==> Label data')
channel = 'depth' 
    local trainMean = trainData.y[{ {},i }]:mean()
    local trainStd = trainData.y[{ {},i }]:std()
    
    local valMean = valData.y[{ {},i }]:mean()
    local valStd = valData.y[{ {},i }]:std()
    
    local testMean = testData.y[{ {},i }]:mean()
    local testStd = testData.y[{ {},i }]:std()
    
    print('training data, '..channel..'-channel, mean: ' .. trainMean)
    print('training data, '..channel..'-channel, standard deviation: ' .. trainStd)
    
    print('test data, '..channel..'-channel, mean: ' .. valMean)
    print('test data, '..channel..'-channel, standard deviation: ' .. valStd)
    
    print('test data, '..channel..'-channel, mean: ' .. testMean)
    print('test data, '..channel..'-channel, standard deviation: ' .. testStd)

    print('label mean: '..l_mean..' std: ' .. l_std)
    
----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> visualizing data (Skipping)')

-- Visualization is quite easy, using image.display(). Check out:
-- help(image.display), for more info about options.

if false then
   local first256Samples_y = {trainData.x[{ {5},1 }],trainData.y[{ {5},1 }]}
   image.display{image=first256Samples_y, nrow=16, legend='Some training examples: Y channel'}
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