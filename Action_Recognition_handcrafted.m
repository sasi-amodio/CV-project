e%   Project AVPR 2022 : First System 
%   Student : Salvatore Davide Amodio

%% Clear Variables, Close Current Figures, and Create Results Directory 
clc;
clear all;
close all;

%% Load Dataset and Split in Train, Validation and Test Set

trainingSet = imageDatastore("Dataset/TrainSet", 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
testSet = imageDatastore("Dataset/TestSet", 'IncludeSubfolders', true,'LabelSource', 'foldernames');

numTrainInstancesForClass = 40;
[trainingSet,validationSet] = splitEachLabel(trainingSet,numTrainInstancesForClass,'randomize');

trainingSize = numel(trainingSet.Files);
validationSize = numel(validationSet.Files);
testSize = numel(testSet.Files);

trainingLabels = trainingSet.Labels;
validationLabels = validationSet.Labels;
testLabels = testSet.Labels;

%% Load Pretrained Network 

net = googlenet();

% extract the layer graph from the trained network and plot the layer graph

inputSize = net.Layers(1).InputSize;

lgraph = layerGraph(net);
%figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
%analyzeNetwork(net)
%plot(lgraph)

%% Replace Final Layers

% remove the last layers 

lgraph = removeLayers(lgraph, {'loss3-classifier','prob','output'});
numClasses = numel(categories(trainingSet.Labels));

% create the appropriate layers

newLayers = [
    fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];

% connect the last transferred layer remaining in the network to the new layers

lgraph = addLayers(lgraph,newLayers);
lgraph = connectLayers(lgraph,'pool5-drop_7x7_s1','fc');

% figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
% plot(lgraph)
% ylim([0,10])

% freeze Initial Layers

layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:110) = freezeWeights(layers(1:110));
lgraph = createLgraphUsingConnections(layers,connections);

%% Train network 

% images in the image datastore may have different sizes. Use an augmented image 
% datastore to automatically resize the training images. 

pixelRange = [-30 30];
scaleRange = [0.9 1.1];

imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange);

augimdsTrain = augmentedImageDatastore(inputSize(1:2),trainingSet, ...
    'DataAugmentation',imageAugmenter);

augimdsValidation = augmentedImageDatastore(inputSize(1:2),validationSet, ...
    'DataAugmentation',imageAugmenter);

options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'ValidationPatience',Inf, ...
    'Verbose',false ,...
    'Plots','training-progress');

net = trainNetwork(augimdsTrain,lgraph,options);

%% Classify the validation images using the fine-tuned network, and calculate the classification accuracy

[trainPred,~] = classify(net,augimdsTrain);
trainAccuracy = mean(trainPred == trainingLabels);

[validationPred,~] = classify(net,augimdsValidation);
validationAccuracy = mean(validationPred == validationLabels);

augimdsTest = augmentedImageDatastore(inputSize(1:2),testSet, ...
    'DataAugmentation',imageAugmenter);

[testPred,probs] = classify(net,augimdsTest);
testAccuracy = mean(testPred == testLabels);

idx = randperm(numel(testSet.Files),7);
figure
for i = 1:7
    subplot(3,3,i)
    I = readimage(testSet,idx(i));
    imshow(I)
    label = testPred(idx(i));
    title(string(label) + ", " + num2str(100*max(probs(idx(i),:)),3) + "%");
end

%% createLgraphUsingConnections

function lgraph = createLgraphUsingConnections(layers,connections)
    lgraph = layerGraph();
    for i = 1:numel(layers)
        lgraph = addLayers(lgraph,layers(i));
    end
    
    for c = 1:size(connections,1)
        lgraph = connectLayers(lgraph,connections.Source{c},connections.Destination{c});
    end
end

%% freezeWeights
function layers = freezeWeights(layers)

    for ii = 1:size(layers,1)
        props = properties(layers(ii));
        for p = 1:numel(props)
            propName = props{p};
            if ~isempty(regexp(propName, 'LearnRateFactor$', 'once'))
                layers(ii).(propName) = 0;
            end
        end
    end
end
