% Assuming centroidPositive and centroidNegative are already calculated

% Calculate the slope of the line joining the centroids
m = (centroidNegative(2) - centroidPositive(2)) / (centroidNegative(1) - centroidPositive(1));

% Midpoint of the centroids (from Step d)
midpoint = (centroidPositive + centroidNegative) / 2;

% Calculate the y-intercept of the line using the midpoint
b = midpoint(2) - m * midpoint(1);

% Display the slope and intercept
disp(['Slope (m) of the line: ', num2str(m)]);
disp(['Intercept (b) of the line: ', num2str(b)]);