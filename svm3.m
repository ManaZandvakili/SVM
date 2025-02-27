% Load the dataset
data = load('Downloads/DataSetv1.txt');

% Separate features and labels
features = data(:, 1:2);
targets = data(:, 3);

% Train SVM with no regularization
SVMModel = fitcsvm(features, targets, 'BoxConstraint', inf,'Standardize',true, 'KernelFunction', 'linear');

% Separate positive and negative examples
positiveExamples = features(targets == 1, :);
negativeExamples = features(targets == -1, :);

% Compute centroids of positive and negative examples
centroidPositive = mean(positiveExamples, 1);
centroidNegative = mean(negativeExamples, 1);

% Compute the midpoint of the line joining the centroids
midpoint = (centroidPositive + centroidNegative) / 2;

% Display centroids and midpoint
disp('Centroid of positive examples:');
disp(centroidPositive);
disp('Centroid of negative examples:');
disp(centroidNegative);
disp('Midpoint of the centroids:');
disp(midpoint);


% Check if the midpoint is on the decision boundary
% The decision boundary equation: w'*x + b = 0
w = SVMModel.Beta;         % Weight vector
b = SVMModel.Bias;         % Bias term
decisionValue = dot(w, midpoint) + b;

if abs(decisionValue) < 1e-6  % Close to zero, allowing a small tolerance
    disp('The midpoint lies on the decision boundary.');
else
    disp('The midpoint does not lie on the decision boundary.');
end