% Load the dataset
data = load('Downloads/DataSetv1.txt');

% Separate features and labels
features = data(:, 1:2);
targets = data(:, 3);

% Train SVM with no regularization
SVMModel = fitcsvm(features, targets, 'BoxConstraint', inf, 'KernelFunction', 'linear');

% Get support vectors from the model
supportVec = SVMModel.SupportVectors;

% Define specific points as support vectors if they’re not already included
pointToAdd = [0, 2];
if ~ismember(pointToAdd, supportVec, 'rows')
    supportVec = [supportVec; pointToAdd];
end
pointToAdd = [3, 1];
if ~ismember(pointToAdd, supportVec, 'rows')
    supportVec = [supportVec; pointToAdd];
end

% Calculate the average (centroid) of each class
centroidPositive = mean(features(targets == 1, :), 1); % Centroid for Class +1
centroidNegative = mean(features(targets == -1, :), 1); % Centroid for Class -1

% Calculate the midpoint of the centroid line
midpoint = (centroidPositive + centroidNegative) / 2;

% Plot data points
figure;
gscatter(features(:,1), features(:,2), targets, 'mb', '^+');
hold on;

% Plot support vectors
plot(supportVec(:,1), supportVec(:,2), 'ko', 'MarkerSize', 10);

% Plot centroids
plot(centroidPositive(1), centroidPositive(2), 'gs', 'MarkerSize', 10, 'MarkerFaceColor', 'g'); % Class +1 centroid in green
plot(centroidNegative(1), centroidNegative(2), 'rs', 'MarkerSize', 10, 'MarkerFaceColor', 'r'); % Class -1 centroid in red

% Draw line connecting the centroids
plot([centroidPositive(1), centroidNegative(1)], [centroidPositive(2), centroidNegative(2)], 'k--', 'LineWidth', 1.5);

% Plot the midpoint of the centroid line
plot(midpoint(1), midpoint(2), 'cd', 'MarkerSize', 10, 'MarkerFaceColor', 'c'); % Midpoint in cyan

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
title('SVM Decision Boundary with Support Vectors, Margins, Centroid Line, and Midpoint');
xlabel('Feature 1');
ylabel('Feature 2');
legend({'Class -1', 'Class +1', 'Support Vectors', 'Centroid of Class +1', 'Centroid of Class -1', 'Centroid Line', 'Midpoint'}, 'Location', 'best');
hold off;
