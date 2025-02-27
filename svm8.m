% Load the added dataset
addedData = load('Desktop/NewDataSetv1.txt');

% Separate features and labels
features = addedData(:, 1:2);
targets = addedData(:, 3);

% Train SVM on the new dataset with regularization
newSVMModel = fitcsvm(features, targets, 'BoxConstraint', 0.0000001,'KernelFunction', 'linear');

% Plot the added data
figure;
gscatter(features(:,1), features(:,2), targets, 'mb', '^+');
hold on;

% Highlight the new positive cluster center for reference
plot(40, 50, 'cs', 'MarkerSize', 10, 'MarkerFaceColor', 'c'); % Cluster center

% Plot the new decision boundary and margins
xlim = get(gca, 'XLim');
ylim = get(gca, 'YLim');
[x, y] = meshgrid(linspace(xlim(1), xlim(2), 100), linspace(ylim(1), ylim(2), 100));
xy = [x(:), y(:)];

% Get the decision scores for each point in the grid for the new model
[~, score] = predict(newSVMModel, xy);
scoreGrid = reshape(score(:, 2), size(x));
contour(x, y, scoreGrid, [0 0], 'b', 'LineWidth', 2);  % New decision boundary
contour(x, y, scoreGrid, [-1 1], '--b', 'LineWidth', 1); % New margins

% Plot the original decision boundary (if saved from Step c)
%Here we assume that `SVMModel` from Step c is already available in the workspace
[~, origScore] = predict(SVMModel, xy);
origScoreGrid = reshape(origScore(:, 2), size(x));
contour(x, y, origScoreGrid, [0 0], 'k', 'LineWidth', 2);  % Original decision boundary
contour(x, y, origScoreGrid, [-1 1], '--k', 'LineWidth', 1); % Original margins

% Labels and legend
title('Comparing of Decision Boundaries for Original and Added Datasets');
xlabel('Feature 1');
ylabel('Feature 2');
legend({'Class +1', 'Class -1', 'Cluster Center (40, 50)', 'New Decision Boundary', 'Original Decision Boundary'}, 'Location', 'best');
hold off;
