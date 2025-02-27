% Load the added dataset
addedData = load('Desktop\NewDataSetv1.txt');

% Separate features and labels
features = addedData(:, 1:2);
targets = addedData(:, 3);

% Plot the original and new data points
figure;
gscatter(features(:,1), features(:,2), targets, 'mb', '^+');
hold on;

% Highlight the new positive cluster center
plot(40, 50, 'cs', 'MarkerSize', 10, 'MarkerFaceColor', 'c'); % Cluster center

% Add labels and legend
title('Added Training Set');
xlabel('Feature 1');
ylabel('Feature 2');
legend({'Class -1', 'Class +1', 'Cluster Center (40, 50)'}, 'Location', 'best');
hold off;
