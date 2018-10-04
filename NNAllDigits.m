%%Author: Philip Giles, UCD
%%This code is very similar  to NN2digits.m. The only signifigant
%%difference being the fact that this script trains and predicts for all
%%digits 0-9 not just 2 specified digits. A lot of the comments will not be
%%repeated in this script because the logic is the same, review NN2digits.m
%%for details on the operation of the script

%%As before the first section takes in the images and labels from the files
%%in your work dir, and reshapes the imagees for input to the matrix

numOutputs = 10; %don't change
NumArray = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]; %0-9
LR = 0.02; %Learning Rate

[training_images, training_labels] = readMNIST('MNIST_TRAINING_IMAGES.idx3-ubyte', 'MNIST_TRAINING_LABELS.idx1-ubyte', 60000, 0);
[testing_images, testing_labels]   = readMNIST('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte', 10000, 0);

test_images = reshape(testing_images, size(testing_images, 1)*size(testing_images, 2), size(testing_images, 3));
images = reshape(training_images, size(training_images, 1)*size(training_images, 2), size(training_images, 3));
%%
%%Training the model
numInputs = size(images,1);

weights = rand(numInputs, numOutputs);
output = zeros(1,numOutputs);
target_output = zeros(1,numOutputs);
error =  zeros(1,numOutputs);

numSelected = size(training_labels, 1) %This will be 60,000 as all images are now selected

for nimage = 1:1:numSelected
    
    %Setting target output, more efficient way of doing this than NN2digits
    target_output = zeros(1,numOutputs);
    target_output(training_labels(nimage)+1) = 1;

    %Calculating the sum (Z) and the activation ( a = sig(z) ). Then
    %updating the weights according to the error of the output.
    for n = 1:1:numOutputs
        sumNode = 0;
        sumNode = sum(images(:,nimage).*weights(:,n));
        output(n) = sigActivation(sumNode);
        error(n) = target_output(n)-output(n);
        weights(:,n) = weights(:,n) + (images(:,nimage)*error(n)*LR); 
    end
    
    fprintf("Epoch: %d\n", nimage);
    target_output;
    output;
    error;
    
end
%%
%%This section is for evaluating the model. 10,000 testing images are
%%available to test.
numTests = 100;

correctPredictions = 0;
incorrectPredictions = 0;

for nimage = 1:1:numTests
    %Predicting
    for n = 1:1:numOutputs
        sumNode = 0;
        sumNode = sum(test_images(:,nimage).*weights(:,n));
        output(n) = sigActivation(sumNode);
    end
    %The highest activation output is our prediction
    [val id] = max(output);
    
    %Printing the prediction and actual label
    fprintf("Predicted number is %i \n", NumArray(id))
    fprintf("The actual number is %i \n", testing_labels(nimage))
    
    if NumArray(id) == testing_labels(nimage)
        correctPredictions = correctPredictions + 1;
    else
        incorrectPredictions = incorrectPredictions + 1;
        %Uncomment if you want to display all incorrect predictions
%         figure;
%         imshow(testing_images(:,:,nimage));
%         title(['Predicted ', num2str(NumArray(id)), ' but Actual is ', num2str(testing_labels(nimage))]);
    end
    
end

accuracy = (correctPredictions)/(correctPredictions+incorrectPredictions)
            
    
    

