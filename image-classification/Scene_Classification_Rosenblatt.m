%% Scene classification using Rosenblatt's perceptron - Part (a) and normalization comparison

clear;
close all;

% Constants for image processing
IMAGE_WIDTH = 32;
IMAGE_HEIGHT = 32;
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT; % Size after conversion to grayscale and reshaping

% Matric number to determine group ID
matricNumber = 'A0280070R'; 
groupID = mod(70, 3);

% Define dataset paths
basePath = fullfile('M:\OneDrive - National University of Singapore\NUS\Sem 2\Machine Learning\Assignments\Assignment 2', sprintf('group_%d', groupID));

animalFolder = {'cat', 'dog', 'deer'}; % Depending on group ID
objectFolder = {'airplane', 'automobile', 'ship'}; % Depending on group ID

trainImages = [];
trainLabels = [];
testImages = [];
testLabels = [];

folders = {animalFolder{groupID+1}, objectFolder{groupID+1}};

for fIdx = 1:numel(folders)
    folder = fullfile(basePath, folders{fIdx});
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

% Add bias unit of 1 to the beginning of each image vector
train_images = [ones(1, size(trainImages, 2)); trainImages];
test_images = [ones(1, size(testImages, 2)); testImages];

%% Train Rosenblatt's perceptron with original data
W = trainPerceptron(train_images, trainLabels);
original_train_accuracy = evaluatePerceptron(W, train_images, trainLabels);
original_test_accuracy = evaluatePerceptron(W, test_images, testLabels);

fprintf('Original Training Accuracy: %.2f%%\n', original_train_accuracy * 100);
fprintf('Original Test Accuracy: %.2f%%\n', original_test_accuracy * 100);

% Combine training and test images for global statistics calculation
allImages = [trainImages, testImages];

% Calculate global mean and standard deviation
globalMean = mean(allImages, 2);
globalStd = std(allImages, 0, 2);

% Normalize both training and test images using global statistics
normalized_train_images = (trainImages - globalMean) ./ globalStd;
normalized_test_images = (testImages - globalMean) ./ globalStd;

% Train Rosenblatt's perceptron using normalized training data
W_normalized = trainPerceptron(normalized_train_images, trainLabels);

% Evaluate on normalized training data
normalized_train_accuracy = evaluatePerceptron(W_normalized, normalized_train_images, trainLabels);

% Evaluate on normalized test data
normalized_test_accuracy = evaluatePerceptron(W_normalized, normalized_test_images, testLabels);

fprintf('Normalized Training Accuracy: %.2f%%\n', normalized_train_accuracy * 100);
fprintf('Normalized Test Accuracy: %.2f%%\n', normalized_test_accuracy * 100);

%% Helper functions

function W = trainPerceptron(images, labels)
    imageSize = size(images, 1); 
    W = rand(imageSize, 1) - 0.5; % Initialize weights
    learningRate = 0.1;
    for epoch = 1:10000 
        shuffleIndices = randperm(size(images, 2));
        errors = 0;
        for i = shuffleIndices 
            output = double((W' * images(:, i)) >= 0);
            error = labels(i) - output;
            W = W + learningRate * error * images(:, i);
            errors = errors + abs(error);
        end
        if errors == 0
            break; % If no errors, perceptron has converged
        end
    end
end

function accuracy = evaluatePerceptron(W, images, labels)
    outputs = double((W' * images) >= 0);
    accuracy = sum(outputs' == labels) / length(labels); % Convert to proportion
end
