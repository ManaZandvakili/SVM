% Load the original dataset
originalData = load('Downloads/DataSetv1.txt');

% Define the cluster center and number of new samples
clusterCenter = [40.0, 50.0];
numSamples = 1000;

% Generate random variations in the range [-2, 2] for each coordinate
randomOffsets = (rand(numSamples, 2) * 4) - 2;  % Generates values in [-2, 2]

% Create new positive examples by adding the random offsets to the cluster center
newPositiveExamples = clusterCenter + randomOffsets;

% Assign label +1 to each of the new examples
newtargets = ones(numSamples, 1);

% Combine the new examples with targets
newData = [newPositiveExamples, newtargets];

% Append the new data to the original data
addedData = [originalData; newData];

% Save the new dataset to a new file
save('NewDataSetv1.txt', 'addedData', '-ascii');