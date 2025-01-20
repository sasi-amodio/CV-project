%   Project AVPR 2022 : Second System 
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

trainingLabels = trainingSet.Labels;
validationLabels = validationSet.Labels;

%% GridMap 

bestCellSize = -1;
bestTrainingAccuracy = [];
bestValidationAccuracy = -1;
bestSVM_C = [];
bestKernel = [];
bestMdl = [];

for cellSize = [4,8,16,32,64,128]

    % Initializiation of matrices that are going to contain feature vectors 

    disp("starts feature extraction");

    img = readimage(trainingSet, 1);
    img = rgb2gray(img);
    img = imgaussfilt(img);
    img = imresize(img, [256 256]);
    img = histeq(img);
    
    feature_vector = extractULDPFeatures(img,cellSize);
    lbpFeatureSize = length(feature_vector);

    trainingFeatures = zeros(trainingSize, lbpFeatureSize, 'single');
    validationFeatures = zeros(validationSize, lbpFeatureSize, 'single');
    
    % feature extractions 

    for i = 1 : trainingSize
        img = readimage(trainingSet, i);
        img = rgb2gray(img);
        img = imgaussfilt(img);
        img = imresize(img, [256 256]);
        img = histeq(img);
        
        trainingFeatures(i, :) = extractULDPFeatures(img, cellSize);
    end
    
    for i = 1 : validationSize
        img = readimage(validationSet, i);
        img = rgb2gray(img);
        img = imgaussfilt(img);
        img = imresize(img, [256 256]);
        img = histeq(img);
        
        validationFeatures(i, :) = extractULDPFeatures(img, cellSize);
    end
    

    % find the best parameters combinations  
        
    for SVM_C = [0.001, 0.05, 0.01, 0.5, 0.1, 1, 3, 5] 
        for SVM_Kernel = ["polynomial", "linear", "rbf"] 

            disp("starts training");

            opts = templateSVM('KernelFunction', SVM_Kernel, 'BoxConstraint', SVM_C, 'kernelScale', 'auto');
            Mdl = fitcecoc(trainingFeatures,trainingLabels,'Learners',opts);

            % evaluate the classifier on training data

            trainingPred = predict(Mdl, trainingFeatures);
            trainingAccuracy = mean(trainingPred == trainingLabels);
        
            % evaluate the classifier on validation data

            validationPred = predict(Mdl, validationFeatures);
            validationAccuracy = mean(validationPred == validationLabels);
               

            % choose the best model by comparing the validation accuracy

            if validationAccuracy > bestValidationAccuracy 
                bestSVM_C = SVM_C;
                bestKernel = SVM_Kernel;
                bestTrainingAccuracy = trainingAccuracy;
                bestValidationAccuracy = validationAccuracy;  
                bestMdl = Mdl;
                bestCellSize = cellSize;

                disp("partial result : SVM_C = "+ SVM_C + " , Kernel = " + SVM_Kernel + ...
                    ", TrainingAccuracy = " + trainingAccuracy + ...
                    ", ValidationAccuracy = " + validationAccuracy + ", cellSize = " + cellSize);
            end 
        end
    end 
end


%% Evaluate the performance of the best classifier computed

testSize = numel(testSet.Files);
testLabels = testSet.Labels;

% Initializiation of matrix that is going to contain feature vectors 

img = readimage(testSet, 1);
img = rgb2gray(img);
img = imgaussfilt(img);
img = imresize(img, [256 256]);
img = histeq(img);

feature_vector = extractULDPFeatures(img, cellSize);
lbpFeatureSize = length(feature_vector);

testFeatures = zeros(testSize, lbpFeatureSize, 'single');

% feature extractions 

for i = 1 : testSize
    img = readimage(testSet, i);
    img = rgb2gray(img);
    img = imgaussfilt(img);
    img = imresize(img, [256 256]);
    img = histeq(img);
    
    testFeatures(i, :) = extractULDPFeatures(img, cellSize);
end

% evaluate the classifier on test data

testPred = predict(bestMdl, testFeatures);
testAccuracy = mean(testPred == testLabels);

disp("final result : SVM_C = "+ bestSVM_C + " , Kernel = " + bestKernel + ...
    ", Gamma = 'auto' , trainingAccuracy = " + bestTrainingAccuracy + ...
    ", testAccuracy = " + testAccuracy + ", cellSize = " + bestCellSize);



%% extractULDPFeatures


function featureVector = extractULDPFeatures(img,subregions)

    bins = 59;
    featureVector = [];

    % "uniformPatterns" stores all the possible uniformPatterns 
    % (58 with 8-bit binary numbers) in decimal numbering system
    % (considering a range [1,256]). The index of the array represents the 
    % label associated with that uniform pattern.

    % Es. uniformPattern = '0001110' is equal to '14 + 1' (remember that 
    % the range is [1,256]) and is stored as '15' in the 10-th position of 
    % uniformPatterns

    uniformPatterns = getUniformPatterns();

    % all the pixels that do not match with any uniform pattern will get 59
    % as a label.
    nonUniformPatternBinID = 59;

    % Kirsch compass masks
    northDirectionMask      =  [-3 -3  5; -3  0  5; -3 -3  5];
    northWestDirectionMask  =  [-3  5  5; -3  0  5; -3 -3 -3]; 
    westDirectionMask       =  [ 5  5  5; -3  0 -3; -3 -3 -3];
    southWestDirectionMask  =  [ 5  5 -3;  5  0 -3; -3 -3 -3];
    southDirectionMask      =  [ 5 -3 -3;  5  0 -3;  5 -3 -3];
    southEastDirectionMask  =  [-3 -3 -3;  5  0 -3;  5  5 -3];
    eastDirectionMask       =  [-3 -3 -3; -3  0 -3;  5  5  5];
    northEastDirectionMask  =  [-3 -3 -3; -3  0  5; -3  5  5];
       
    [N,M] = size(img);
    subN = int16(N/sqrt(subregions));
    subM = int16(M/sqrt(subregions));
    
    % zero padding operation     
    img = vertcat(zeros(1,M),img);
    img = vertcat(img,zeros(1,M));
    img = horzcat(img,zeros(N+2,1));
    img = horzcat(zeros(N+2,1),img);
    
    for h=1: sqrt(subregions)
        for k=1: sqrt(subregions)
           
            newSubRegion = zeros(subN,subM,'int16');
    
            % the range of i and j take into account the padding operation
            for i = ((h-1) * subN) +2: (h * subN) +1 
                for j = ((k-1) * subM) +2 : (k * subM) +1
                    
                    window = img(i-1:i+1,j-1:j+1);
            
                    % compute the edge responses using the eight Kirsch compass masks
                    m2 = sum(double(window) .* northDirectionMask,      'all' ); 
                    m3 = sum(double(window) .* northWestDirectionMask,  'all' );
                    m1 = sum(double(window) .* northEastDirectionMask,  'all' );
                    m0 = sum(double(window) .* eastDirectionMask,       'all' );
                    m4 = sum(double(window) .* westDirectionMask,       'all' );
                    m7 = sum(double(window) .* southEastDirectionMask,  'all' );
                    m5 = sum(double(window) .* southWestDirectionMask,  'all' );
                    m6 = sum(double(window) .* southDirectionMask,      'all' );
            
                    %    newWindow = [m3 > 0, m2 > 0, m1 > 0; 
                    %                 m4 > 0,   0   , m0 > 0; 
                    %                 m5 > 0, m6 > 0, m7 > 0];
               
                   uldpCode = strcat(num2str(m7 > 0),  num2str(m6 > 0),  num2str(m5 > 0), num2str(m4 > 0),  ...
                                     num2str(m3 > 0), num2str(m2 > 0),  num2str(m1 > 0),  num2str(m0 > 0));
     
                   % each possible uniform pattern has an ID associated which
                   % corrisponds at the index of "uniformPatterns" array 
                   pattern = bin2dec(uldpCode) + 1;
    
                   % return the uniformpattern index if pattern is an
                   % uniformpattern 
                   uniformPatternBinID = find(uniformPatterns==pattern);
                   
                   partial_i = i - ((h-1) * subN) - 1;
                   partial_j = j - ((k-1) * subM) - 1;

                   if isempty(uniformPatternBinID)
                       newSubRegion(partial_i,partial_j) = nonUniformPatternBinID;
                   else
                       % each pixel that does not match with any uniform pattern 
                       % will get the same label = nonUniformPatternBinID (59) 
                       newSubRegion(partial_i,partial_j) = uniformPatternBinID;
                   end
                end 
            end
            
            % "flattenedSubRegion" contains all the values present into the newSubRegion 
            flattenedSubRegion = newSubRegion(:);
    
            subRegionHistogram = histogram(flattenedSubRegion, BinEdges=1:bins+1, Normalization="probability");

            % return the histogram as a numeric vector.
            subRegionVector = subRegionHistogram.Values;

            % for plotting 
            % title(['ULDP features histogram of subregion ',num2str(h+k-1)])
            % end  
    
            % concatenate the subRregionVectors 
            if isempty(featureVector) 
                featureVector = subRegionVector;
            else 
                featureVector = [featureVector, subRegionVector];
            end
        end
    end

    % for plotting 
    % bar(featureVector)
    % title('ULDP features histogram before L2-norm')
    % end 

    s = 0;
    for i = 1:size(featureVector,2)
        s = s + featureVector(i)^2;
    end
    
    L2norm = sqrt(s);
    featureVector = featureVector / L2norm;
    
    % for plotting 
    %bar(featureVector)
    %title('ULDP features histogram after L2-norm')
    % end 
end



%% getUniformPatterns

function uniformPatterns = getUniformPatterns() 
    
    % in decimal, from 1 to 256
    uniformPatterns = zeros(1,58,'int16');
    count = 1;
 
    for i= 1 : 256     
        value = dec2bin(i-1,8);
    
        count0 = extract(value, '01');
        count1 = extract(value, '10');
          
        if size(count0,1) + size(count1,1) <= 2 
            value = bin2dec(value) + 1;
            uniformPatterns(count) = value;
            count = count +1;
        end
    end
end