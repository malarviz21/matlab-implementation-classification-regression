clear; close all;

% Constants for image processing
IMAGE_WIDTH = 32;
IMAGE_HEIGHT = 32;
NUM_CLASSES = 2; % For binary classification

% Dataset paths setup
matricNumber = 'A0280070R'; % Example matric number
groupID = mod(70, 3); % Calculate group ID
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

% Reshape for MATLAB Neural Network which expects inputs as columns
trainImages = reshape(trainImages, IMAGE_WIDTH*IMAGE_HEIGHT, []);
testImages = reshape(testImages, IMAGE_WIDTH*IMAGE_HEIGHT, []);

% Initialize variables for plotting
numEpochs = 1000; % Define the number of epochs
trainAccHistory = zeros(1, numEpochs);
testAccHistory = zeros(1, numEpochs);

% Create the pattern recognition network
hiddenLayerSize = 20;
net = patternnet(hiddenLayerSize);

% Setup division of data - Assuming the default division or custom division here
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% Prepare for manual epoch control
net.trainParam.epochs = 1; % Train for 1 epoch at a time
net.trainParam.showWindow = false; % Optionally hide the training window to speed up

for epoch = 1:numEpochs
    % Train the network for one epoch
    [net, tr] = train(net, trainImages, trainLabels');
    
    % Compute training accuracy
    trainPred = net(trainImages) > 0.5; % For binary classification
    trainAccuracy = sum(trainPred == trainLabels') / numel(trainLabels) * 100;
    trainAccHistory(epoch) = trainAccuracy;
    
    % Compute testing accuracy
    testPred = net(testImages) > 0.5; % For binary classification
    testAccuracy = sum(testPred == testLabels') / numel(testLabels) * 100;
    testAccHistory(epoch) = testAccuracy;
    
    fprintf('Epoch %d/%d - Training Accuracy: %.2f%%, Testing Accuracy: %.2f%%\n', epoch, numEpochs, trainAccuracy, testAccuracy);
end

% Plotting training epochs vs. accuracy
figure;
plot(1:numEpochs, trainAccHistory, 'b-', 'LineWidth', 2);
hold on;
plot(1:numEpochs, testAccHistory, 'r-', 'LineWidth', 2);
title('Training and Testing Accuracy over Epochs');
xlabel('Epoch');
ylabel('Accuracy (%)');
legend('Training Accuracy', 'Testing Accuracy');
grid on;

% Calculate and display Accuracy
trainAccuracy = sum(trainPred == trainLabels') / numel(trainLabels) * 100;
testAccuracy = sum(testPred == testLabels') / numel(testLabels) * 100;

fprintf('Training Set Accuracy: %.2f%%\n', trainAccuracy);
fprintf('Testing Set Accuracy: %.2f%%\n', testAccuracy);