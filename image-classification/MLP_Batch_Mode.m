%% MLP for Image Classification - Batch Mode Training

clear;
close all;

% Image dimensions
IMAGE_WIDTH = 32;
IMAGE_HEIGHT = 32;
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT;

% Determine group based on matriculation number
matricNumber = 'A0280070R'; % Placeholder, replace with your actual matric number
groupID = mod(70, 3);

% Define dataset directory
baseDir = fullfile('M:\OneDrive - National University of Singapore\NUS\Sem 2\Machine Learning\Assignments\Assignment 2\', sprintf('group_%d', groupID));

animalFolder = {'cat', 'dog', 'deer'}; % Depending on group ID
objectFolder = {'airplane', 'automobile', 'ship'}; % Depending on group ID

trainImages = [];
trainLabels = [];
testImages = [];
testLabels = [];

folders = {animalFolder{groupID+1}, objectFolder{groupID+1}};

for fIdx = 1:numel(folders)
    folder = fullfile(baseDir, folders{fIdx});
    for idx = 0:499
        imgFile = fullfile(folder, sprintf('%03d.jpg', idx));
        img = imread(imgFile);
        imgGray = rgb2gray(img); % Convert to grayscale
        imgVector = reshape(imgGray, [], 1); % Reshape into vector
        
        if idx < 450 % Training set
            trainImages = [trainImages, double(imgVector)];
            trainLabels = [trainLabels; fIdx-1];
        else % Test set
            testImages = [testImages, double(imgVector)];
            testLabels = [testLabels; fIdx-1];
        end
    end
end

% Normalize the image vectors to range [0, 1]
trainImages = trainImages / 255.0;
testImages = testImages / 255.0;

% Define the MLP
hiddenLayerSize = [100]; % One hidden layer
trainAlgorithm = 'traincgb'; % Consider experimenting with others, e.g., 'trainlm'
performParam = 'mse';
net = patternnet(hiddenLayerSize, trainAlgorithm, performParam);

% Adjust regularization (for Ques 3(d))
net.performParam.regularization = 0.25; % Adjust based on trial and error, was 0.1

% Setup Division of Data
net.divideFcn = 'divideblock'; % Divide data block-wise
net.divideParam.trainRatio = 0.75;
net.divideParam.valRatio = 0.15;
net.divideParam.testRatio = 0.10; % Adjusted for a slight increase in test set size

% Training Parameters
net.trainParam.epochs = 100;
net.trainParam.goal = 1e-4;
net.trainParam.max_fail = 10; % Allowing more failures before early stopping
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'logsig';

% Correcting label encoding for binary classification
% Ensure labels are 0 and 1 if not already
% Adjust if using different encoding

% Train the MLP
[net, tr] = train(net, trainImages, trainLabels');

% Evaluate the MLP
trainPred = round(net(trainImages));
trainAcc = sum(trainPred' == trainLabels) / numel(trainLabels) * 100; % Ensure predictions are correctly compared
testPred = round(net(testImages));
testAcc = sum(testPred' == testLabels) / numel(testLabels) * 100; % Ensure predictions are correctly compared

fprintf('Training Accuracy: %.2f%%\n', trainAcc);
fprintf('Test Accuracy: %.2f%%\n', testAcc);