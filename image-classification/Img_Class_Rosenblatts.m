%% Image classification using Rosenblatt’s perceptron

clear;
close all;

% Constants
IMAGE_WIDTH = 32;
IMAGE_HEIGHT = 32;
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT; % For 32x32 grayscale images

% Determine group based on matric number
matricNumber = 'A0280070R';
groupID = mod(70, 3);

% Print the groupID
fprintf('The group ID assigned based on the matric number %s is: %d\n', matricNumber, groupID);

% Define paths
basePath = fullfile('M:\OneDrive - National University of Singapore\NUS\Sem 2\Machine Learning\Assignments\Assignment 2', sprintf('group_%d', groupID));

animalFolder = {'cat', 'dog', 'deer'}; % Depending on group ID
objectFolder = {'airplane', 'automobile', 'ship'}; 

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
trainImages = [ones(1, size(trainImages, 2)); trainImages];
testImages = [ones(1, size(testImages, 2)); testImages];

%% Train Rosenblatt's Perceptron with Wrong Label Count Tracking
W = rand(IMAGE_SIZE + 1, 1); % Initialize weights
learningRate = 0.1;
epochs = 0;
converged = false;

maeHistory = []; % To store the mean absolute error for each epoch

% Training loop
epochs = 0;
converged = false;
while ~converged
    totalError = 0; % Accumulate total error for this epoch
    
    for i = 1:size(trainImages, 2)
        % Compute perceptron output
        output = (W' * trainImages(:, i)) >= 0;
        % Calculate error
        error = trainLabels(i) - output;
        % Update weights
        W = W + learningRate * error * trainImages(:, i);
        % Accumulate absolute error
        totalError = totalError + abs(error);
    end
    
    % Calculate and store MAE for this epoch
    mae = totalError / size(trainImages, 2);
    maeHistory = [maeHistory, mae];
    
    % Increment epoch count
    epochs = epochs + 1;
    
    % Check for convergence (no errors)
    if mae == 0
        converged = true;
    end
end

%% Evaluation on Trained Perceptron
% Calculate and display training accuracy
trainOutputs = (W' * trainImages) >= 0;
trainWrongLabels = sum(abs(trainOutputs' - trainLabels));
trainAccuracy = (1 - trainWrongLabels / numel(trainLabels)) * 100;
fprintf('Training Accuracy: %.2f%%\n', trainAccuracy);

% Calculate and display test accuracy
testOutputs = (W' * testImages) >= 0;
testWrongLabels = sum(abs(testOutputs' - testLabels));
testAccuracy = (1 - testWrongLabels / numel(testLabels)) * 100;
fprintf('Test Accuracy: %.2f%%\n', testAccuracy);

% Plot MAE vs. epochs
figure;
plot(1:epochs, maeHistory);
xlabel('Epochs');
ylabel('Mean Absolute Error (MAE)');
title('MAE vs. Epochs during Training');
grid on;
