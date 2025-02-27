% Load the dataset
data = load('Downloads/DataSetv1.txt');

% Separate features and targets
features = data(:, 1:2);
targets = data(:, 3);

% Train SVM with no regularization
SVMModel = fitcsvm(features, targets, 'BoxConstraint', Inf, 'KernelFunction', 'linear');

% Display the model details
disp(SVMModel);
