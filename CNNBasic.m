imds = imageDatastore('change_inner_radial','IncludeSubfolders',true,'LabelSource','foldernames');
labelCount = countEachLabel(imds);
[imdsTrainAndValidation,imdsTest] = splitEachLabel(imds,.8,'randomize');
[imdsTrain,imdsValidation] = splitEachLabel(imdsTrainAndValidation,.7,'randomize');
img = readimage(imds,1);
imgsize = size(img);
layers = [
    imageInputLayer([imgsize(1) imgsize(1) 3])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(4)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',20, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',true, ...
    'Plots','training-progress');

net = trainNetwork(imdsTrain,layers,options);

preds = classify(net, imdsTest);
actual = imdsTest.Labels;
numCorrect = nnz(preds == actual);
accuracy = numCorrect/length(preds);
confusionchart(actual,preds)