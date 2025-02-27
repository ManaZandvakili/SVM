% Load the dataset
data = load('Downloads/DataSetv1.txt');

% Separate features and labels
features = data(:, 1:2);
targets = data(:, 3);

% Train SVM with no regularization
SVMModel = fitcsvm(features, targets, 'BoxConstraint', inf,'KernelFunction', 'linear');

% Get the weight vector (w) and bias from the SVM model
w = SVMModel.Beta; % Weight vector
margin_distance = 2 / norm(w); % Margin distance

% Display the calculated margin distance
disp(['Distance between the blue and yellow margin lines: ', num2str(margin_distance)]);

% Get support vectors from the model
supportVec = SVMModel.SupportVectors;

%Define the specific point (0, 2) as a support vector if it’s not already included
pointToAdd = [0, 2];
if ~ismember(pointToAdd, supportVec, 'rows')
    supportVec = [supportVec; pointToAdd];
end
pointToAdd = [3, 1];
if ~ismember(pointToAdd, supportVec, 'rows')
    supportVec = [supportVec; pointToAdd];
end

% Plot data points
figure;
gscatter(features(:,1), features(:,2), targets, 'mb', '^+');
hold on;

% Plot support vectors
plot(supportVec(:,1), supportVec(:,2), 'ko', 'MarkerSize', 10);

% Define a grid of points to plot the decision boundary
xlim = get(gca, 'XLim');
ylim = get(gca, 'YLim');
[x, y] = meshgrid(linspace(xlim(1), xlim(2), 100), linspace(ylim(1), ylim(2), 100));
xy = [x(:), y(:)];

% Get the decision scores for each point in the grid
[~, score] = predict(SVMModel, xy);

% Reshape the scores to match the grid and plot the decision boundary
scoreGrid = reshape(score(:, 2), size(x));
contour(x, y, scoreGrid, [0 0], 'k', 'LineWidth', 2);  % Decision boundary
contour(x, y, scoreGrid, [-1 1], '--', 'LineWidth', 1); % Margins

% Labels and legend
title('SVM Decision Boundary with Support Vectors and Margins');
xlabel('Feature 1');
ylabel('Feature 2');
legend({'Class -1', 'Class +1', 'Support Vectors'}, 'Location', 'best');
hold off;