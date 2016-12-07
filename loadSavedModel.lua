local mnist = require 'mnist';
local mnist = require 'mnist';

local trainData = mnist.traindataset().data:float();
local trainLabels = mnist.traindataset().label:add(1);
testData = mnist.testdataset().data:float();
testLabels = mnist.testdataset().label:add(1);

--We'll start by normalizing our data
local mean = trainData:mean()
local std = trainData:std()
trainData:add(-mean):div(std); 
testData:add(-mean):div(std);


----- ### Shuffling data

function shuffle(data, labels) --shuffle data function
    local randomIndexes = torch.randperm(data:size(1)):long() 
    return data:index(1,randomIndexes), labels:index(1,randomIndexes)
end

------   ### Define model and criterion

require 'nn'
require 'cunn'


local inputSize = 28*28
local outputSize = 10
local layerSizeOptions = {{inputSize,64,64}}
local batchSizes = {16}
local totalEpochs = {30}
local learningRates = {0.01}



function loadModel(filename)
	return torch.load(filename)
end


for layerOptionIndex = 1,#layerSizeOptions do
	for batchIndex =1, #batchSizes do
		for epochIndex =1, #totalEpochs do
			for learningRateIndex =1, #learningRates do
				sumOfError = 0
				for iteration =1 ,1 do	
					print("global model iteration number "..iteration)
					model = nn.Sequential()
					model = loadModel("modelParamsInFile.txt")
					model:cuda()
				
					---- ### Classification criterion

					criterion = nn.CrossEntropyCriterion():cuda()


					require 'optim'
					batchSize = batchSizes[batchIndex]

					optimState = {
						learningRate = learningRates[learningRateIndex]
					}

					--- ### Main evaluation + training function

					function forwardNet(data, labels, train)
						timer = torch.Timer()
						--another helpful function of optim is ConfusionMatrix
						local confusion = optim.ConfusionMatrix(torch.range(0,9):totable())
						local lossAcc = 0
						local numBatches = 0
						for i = 1, data:size(1) - batchSize, batchSize do
							numBatches = numBatches + 1
							local x = data:narrow(1, i, batchSize):cuda()
							local yt = labels:narrow(1, i, batchSize):cuda()
							local y = model:forward(x)
							local err = criterion:forward(y, yt)
							lossAcc = lossAcc + err
							confusion:batchAdd(y,yt)
						end
						
						confusion:updateValids()
						local avgLoss = lossAcc / numBatches
						local avgError = 1 - confusion.totalValid
						print(timer:time().real .. ' seconds')
						return avgLoss, avgError, tostring(confusion)
					end



					epochs = totalEpochs[epochIndex]

					testLoss = torch.Tensor(epochs)
					testError = torch.Tensor(epochs)

					for e = 1, epochs do
						testLoss[e], testError[e], confusion = forwardNet(testData, testLabels, false)
						
						if e % 5 == 0 then
							print('Epoch ' .. e .. ':')
							print('Test error: ' .. testError[e], 'Test Loss: ' .. testLoss[e])
							print(confusion)
						end
					end


					optimState = {
						learningRate = learningRates[learningRateIndex],
						momentum = 0.9,
						weightDecay = 1e-3
						
					}
					for e = 1, epochs do
						testLoss[e], testError[e], confusion = forwardNet(testData, testLabels, false)
					end
					
					print('Test error: ' .. testError[epochs], 'Test Loss: ' .. testLoss[epochs])
				end
			end
		end
	end				
end












