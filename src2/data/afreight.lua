require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nnx'      -- provides a normalization operator
require 'nn'

local data = {}
data.__index = data
function data.create(n_sequence, width, height, batch_size) 
	----------------------------------------------------------------------
	print(sys.COLORS.red ..  '==> loading dataset')

	local n = n_sequence
	-- We load the dataset from disk
	local imagesAll = torch.Tensor(n,3,height,width)
	local labelsAll = torch.Tensor(n,3,height,width)
	
	-- classes: GLOBAL var!
	
	
	-- load backgrounds:
	for f=1,n do
		local number = f
	
		if f<10 then
			number = '00'..f
		else if f<100 then
			number = '0'..f
		end
		end
	
	  	imagesAll[f] = image.load('../datasets/afreightdata/data/afreightim'..number..'.png') 
	  	labelsAll[f] = image.load('../datasets/afreightdata/label/afreightseg'..number..'.png') 
	end
	
	-- shuffle dataset: get shuffled indices in this variable:
	local labelsShuffle = torch.randperm((#labelsAll)[1])
	
	local portionTrain = 0.55 -- 55% is train data
	local portionValid = 0.33 -- 33% is valid data, rest is test data
	
	local trsize = torch.floor(labelsShuffle:size(1)*portionTrain)
	local valsize = torch.floor(labelsShuffle:size(1)*portionValid)
	local tesize = labelsShuffle:size(1) - trsize - valsize
	
	-- create train set:
	trainData = {
	   data = torch.Tensor(trsize, 3, height, width),
	   labels = torch.Tensor(trsize, 1, height, width ),
	   size = function() return trsize end,
	   pos = 0
	}
	
	--create validation set:
	testData = {
	    data = torch.Tensor(valsize, 3, height, width ),
	    labels = torch.Tensor(valsize, 1,height, width ),
	    size = function() return valsize end,
	    pos = 0
	}
	
	--create test set:
	valData = {
	    data = torch.Tensor(tesize, 3, height, width),
	    labels = torch.Tensor(tesize, 1, height, width),
	    size = function() return tesize end,
	    pos = 0
	}
	
	function label(y) 
		
		local timer = torch.Timer()
		print(sys.COLORS.red ..  '==> labeling')
		torch.mul(y,y,255)
		torch.floor(y,y)
		
		labels = {}
		labels_rev = {}
		local new_y = torch.Tensor(y:size(1), 1, height, width ):zero()
		local index = 1
	
		for i = 1, y:size(1) do
		    for x = 1, y:size(3) do 
		    	for yt = 1, y:size(4) do
		        	local id = ''..y[i][1][x][yt]..'.'..y[i][2][x][yt]..'.'..y[i][3][x][yt]
		        	if labels[id] then
		        	    new_y[i][1][x][yt] = labels[id]
		        	else
		        	    labels[id] = index
		        	    labels_rev[index] = {}
		        	    labels_rev[index][1] = y[i][1][x][yt]
		        	    labels_rev[index][2] = y[i][2][x][yt]
		        	    labels_rev[index][3] = y[i][3][x][yt]
		        	    index = index + 1
		        	    new_y[i][1][x][yt] = labels[id]
		        	end
		        end
		    end
		    
		end
		local time = timer:time().real
		print('Passed: '..time)
		collectgarbage();
		
		--new_y = torch.div(new_y,index)
		return new_y
	end
	
	labelsAll = label(labelsAll)

	for i=1,trsize do
	   trainData.data[i] = imagesAll[labelsShuffle[i]]:clone()
	   trainData.labels[i] = labelsAll[labelsShuffle[i]]:clone()
	end
	
	for i=trsize+1,(trsize+valsize) do
	   	testData.data[i-trsize] = imagesAll[labelsShuffle[i]]:clone()
	   	testData.labels[i-trsize] = labelsAll[labelsShuffle[i]]:clone()
	end
	
	for i=trsize+valsize+1,valsize+tesize+trsize do
	   	valData.data[i-trsize-valsize] = imagesAll[labelsShuffle[i]]:clone()
	   	valData.labels[i-trsize-valsize] = labelsAll[labelsShuffle[i]]:clone()
	end
	
	
	-- remove from memory temp image files:
	imagesAll = nil
	labelsAll = nil
	collectgarbage();
	
	
	----------------------------------------------------------------------
	print(sys.COLORS.red ..  '==> preprocessing data')
	-- faces and bg are already YUV here, no need to convert!
	
	-- Preprocessing requires a floating point representation (the original
	-- data is stored on bytes). Types can be easily converted in Torch, 
	-- in general by doing: dst = src:type('torch.TypeTensor'), 
	-- where Type=='Float','Double','Byte','Int',... Shortcuts are provided
	-- for simplicity (float(),double(),cuda(),...):
	
	trainData.data = trainData.data:float()
	valData.data = valData.data:float()
	testData.data = testData.data:float()
	
	trainData.labels = trainData.labels:float()
	valData.labels = valData.labels:float()
	testData.labels = testData.labels:float()
	
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
	
	 for i = 1,trainData:size() do 
	    trainData.data[i] = image.rgb2yuv(trainData.data[i])
	    --trainData.labels[i] = image.rgb2yuv(trainData.labels[i])
	 end
	
	 for i = 1,valData:size() do
	    valData.data[i] = image.rgb2yuv(valData.data[i])
	    --valData.labels[i] = image.rgb2yuv(valData.labels[i])
	 end
	
	 for i = 1,testData:size() do
	    testData.data[i] = image.rgb2yuv(testData.data[i])
	    --testData.labels[i] = image.rgb2yuv(testData.labels[i])
	 end
	
	   
	-- Name channels for convenience
	local channels = {'y'}
	
	-- Normalize each channel, and store mean/std
	-- per channel. These values are important, as they are part of
	-- the trainable parameters. At test time, test data will be normalized
	-- using these values.
	print(sys.COLORS.red ..  '==> preprocessing data: normalize each feature (channel) globally')
	local mean = {}
	local std = {}
	
	local l_mean = {}
	local l_std = {}
	for i,channel in ipairs(channels) do
	   -- normalize each channel globally:
	   mean[i] = trainData.data[{ {},i,{},{} }]:mean()
	   std[i] = trainData.data[{ {},i,{},{} }]:std()
	   trainData.data[{ {},i,{},{} }]:add(-mean[i])
	   trainData.data[{ {},i,{},{} }]:div(std[i])
	
	   --l_mean[i] = trainData.data[{ {},i,{},{} }]:mean()
	   --l_std[i] = trainData.data[{ {},i,{},{} }]:std()
	   --trainData.data[{ {},i,{},{} }]:add(-l_mean[i])
	   --trainData.data[{ {},i,{},{} }]:div(l_std[i])
	end
	
	-- Normalize test data, using the training means/stds
	for i,channel in ipairs(channels) do
	   -- normalize each channel globally:
	   valData.data[{ {},i,{},{} }]:add(-mean[i])
	   valData.data[{ {},i,{},{} }]:div(std[i])
	
	   --valData.labels[{ {},i,{},{} }]:add(-l_mean[i])
	   --valData.labels[{ {},i,{},{} }]:div(l_std[i])
	end
	
	-- Normalize test data, using the training means/stds
	for i,channel in ipairs(channels) do
	   -- normalize each channel globally:
	   testData.data[{ {},i,{},{} }]:add(-mean[i])
	   testData.data[{ {},i,{},{} }]:div(std[i])
	
	   --testData.labels[{ {},i,{},{} }]:add(-l_mean[i])
	   --testData.labels[{ {},i,{},{} }]:div(l_std[i])
	end
	
	
	
	-- Local contrast normalization is needed in the face dataset as the dataset is already in this form:
	print(sys.COLORS.red ..  '==> preprocessing data: normalize all three channels locally')
	
	-- Define the normalization neighborhood:
	local neighborhood = image.gaussian1D(5) -- 5 for face detector training
	
	-- Define our local normalization operator (It is an actual nn module, 
	-- which could be inserted into a trainable model):
	local normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()
	
	-- Normalize all channels locally:
	for c in ipairs(channels) do
	   for i = 1,trainData:size() do
	      trainData.data[{ i,{c},{},{} }] = normalization:forward(trainData.data[{ i,{c},{},{} }])
	      --trainData.labels[{ i,{c},{},{} }] = normalization:forward(trainData.labels[{ i,{c},{},{} }])
	   end
	   for i = 1,valData:size() do
	      valData.data[{ i,{c},{},{} }] = normalization:forward(valData.data[{ i,{c},{},{} }])
	      --valData.labels[{ i,{c},{},{} }] = normalization:forward(valData.labels[{ i,{c},{},{} }])
	   end
	   for i = 1,testData:size() do
	      testData.data[{ i,{c},{},{} }] = normalization:forward(testData.data[{ i,{c},{},{} }])
	      --testData.labels[{ i,{c},{},{} }] = normalization:forward(testData.labels[{ i,{c},{},{} }])
	   end
	end
	
	
	----------------------------------------------------------------------
	print(sys.COLORS.red ..  '==> verify statistics')
	
	-- It's always good practice to verify that data is properly
	-- normalized.
	
	for i,channel in ipairs(channels) do
	   local trainMean = trainData.data[{ {},i }]:mean()
	   local trainStd = trainData.data[{ {},i }]:std()
	
	   local valMean = valData.data[{ {},i }]:mean()
	   local valStd = valData.data[{ {},i }]:std()
	
	   local testMean = testData.data[{ {},i }]:mean()
	   local testStd = testData.data[{ {},i }]:std()
	
	   print('training data, '..channel..'-channel, mean: ' .. trainMean)
	   print('training data, '..channel..'-channel, standard deviation: ' .. trainStd)
	
	   print('test data, '..channel..'-channel, mean: ' .. valMean)
	   print('test data, '..channel..'-channel, standard deviation: ' .. valStd)
	
	   print('test data, '..channel..'-channel, mean: ' .. testMean)
	   print('test data, '..channel..'-channel, standard deviation: ' .. testStd)
	
	end
	
	----------------------------------------------------------------------
	print(sys.COLORS.red ..  '==> visualizing data')
	
	-- Visualization is quite easy, using image.display(). Check out:
	-- help(image.display), for more info about options.
	
	if true then
	   local first256Samples_y = {trainData.data[{ {1},1 }],trainData.labels[{ {1},1 }]}
	   --image.display{image=first256Samples_y, nrow=16, legend='Some training examples: Y channel'}
	   local first256Samples_y = {valData.data[{ {1},1 }],valData.labels[{ {1},1 }]}
	   --image.display{image=first256Samples_y, nrow=16, legend='Some valid examples: Y channel'}
	   local first256Samples_y = {testData.data[{ {1},1 }],testData.labels[{ {1},1 }]}
	   --image.display{image=first256Samples_y, nrow=16, legend='Some testing examples: Y channel'}
	end

	function tablelength(T)
	  local count = 0
	  for _ in pairs(T) do count = count + 1 end
	  return count
	end
	
	self = {}
	setmetatable(self, {__index = data})

	-- Exports
	self.trainData = trainData
	self.testData = testData
	self.valData = valData
	self.mean = mean
	self.std = std
	self.labels = labels
	self.labels_rev = labels_rev
	self.ntrain = trsize
	self.ntest = (#testData.data)[1]

	self.current_batch = 0
	self.evaluated_batches = 0

	self.current_test_batch = 0
	self.evaluated_test_batches = 0 
	
	self.width = width
	self.height = height

	self.batch_size = batch_size
	self.nbatches = trsize / batch_size
	
	self.labels_size = tablelength(labels)

	print(sys.COLORS.green ..  '==> Data Ready')

	function self:next_batch()
		
		self.evaluated_batches = self.evaluated_batches + 1
		batch_x = torch.Tensor(self.batch_size, 3, self.width * self.height)
		batch_y = torch.Tensor(self.batch_size, 1, self.width * self.height)

		for i = 1, batch_size do 
			if(self.current_batch*self.batch_size+i>self.ntrain) then break end
			batch_x[i] = trainData.data[self.current_batch*(self.batch_size)+i]
			batch_y[i] = trainData.labels[self.current_batch*(self.batch_size)+i]
		end
		self.current_batch = self.current_batch +1

    	return batch_x, batch_y
	end

	function self:next_eval_batch()
		
		self.evaluated_test_batches = self.evaluated_test_batches + 1
	
		batch_x = torch.Tensor(self.batch_size, 3, self.width * self.height)
		for i = 1, batch_size do 
			if(self.current_test_batch*(self.batch_size)+i>self.ntest) then break end
			batch_x[i] = testData.data[self.current_test_batch*(self.batch_size)+i]
			batch_y[i] = testData.labels[self.current_test_batch*(self.batch_size)+i]
		end
		self.current_test_batch = self.current_test_batch +1

    	return batch_x, batch_y
	end


	return self
end	


return data