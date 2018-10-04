%%Author: Philip Giles, UCD 
%%In this script I have constructed a Neural Network with no hidden layer.
%%This is part of the learning process for how the mathematics of a neural
%%network work. This version of the code will attempt to differentiate
%%between 2 digits in the MNIST dataset, these digits can be changed by the
%%user. 
%%The network composes simply of the input matix, which is a 784 value
%%input for each image (28*28pixels). Each input connects to the output
%%node and each connection has a weight, which initially is a random value
%%between 1 and 0 and is updated througout the training process.
%%The first section of code will read in the images and labels from the 
%%MNIST data set. The readMNIST is a function that was authored by
%%Siddharth Hegde, found at https://uk.mathworks.com/matlabcentral/fileexchange/27675-read-digits-and-labels-from-mnist-database?focused=5154133&tab=function
%%The files should be in your working directory with the 
%%same name as the filenames below. It will read in 60000 training samples
%%and 10000 testing samples. The script then extracts only the relevant
%%samples from the dataset( the 2 numbers we are differentiating between).
%%The last task in this section is to reshape the 28*28 images to be just
%%784*1 in order to be inputted to the neural network.
%Reading in Images and labels, creating new matrix only for 2's and 8's.

%Two numbers we are differentiating between
Num1 = 2;
Num2 = 7;  
%Learning Rate
LR = 0.02;
numOutputs = 2;  %Dont change

NumArray = [Num1, Num2];  %%Putting the numbers in a vector for code reusability


[training_images, training_labels] = readMNIST('MNIST_TRAINING_IMAGES.idx3-ubyte', 'MNIST_TRAINING_LABELS.idx1-ubyte', 60000, 0);
[testing_images, testing_labels]   = readMNIST('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte', 10000, 0);

%selectRelevantData is a function that will extract only the images
%relating to the 2 digits we are interested in.
[training_images_selected, training_labels_selected] = selectRelevantData(training_images, training_labels, Num1, Num2);
[testing_images_selected, testing_labels_selected] = selectRelevantData(testing_images, testing_labels, Num1, Num2);

%Reshaping the images to be vectors of lenght 784 (28*28)
test_images = reshape(testing_images_selected, size(testing_images_selected, 1)*size(testing_images_selected, 2), size(testing_images_selected, 3));
images = reshape(training_images_selected, size(training_images_selected, 1)*size(training_images_selected, 2), size(training_images_selected, 3));
%%
%%This is the section where the trainig of the model occurs. Each epoch
%%trains only 1 image, so there should be roughly 12,000 epochs (depending
%%on the 2 input numbers that wer selected). 
numInputs = size(images,1); %Should be 784, was originally cropping the images to 20*20, so was 400 before
%%initialising various vectors, particularly the weights which are
%%initialised with random values.
weights = rand(numInputs, numOutputs); 
output = zeros(1,numOutputs);
target_output = zeros(1,numOutputs);
error =  zeros(1,numOutputs);

%Number of epochs is dependent on the number of training images that are
%either num1 or num2.
numSelected = size(training_labels_selected);

for nimage = 1:numSelected
    
    %Setting the target outputs
    if training_labels_selected(nimage) == Num1
        target_output = zeros(1,numOutputs);
        target_output(1) = 1;
    end
    if training_labels_selected(nimage) == Num2
        target_output = zeros(1,numOutputs);
        target_output(2) = 1;
    end
    
    
    %For every output node (2 in this case)
    for n = 1:numOutputs
        sumNode = 0;
        %Sum in this case represents Z
        sumNode = sum(images(:,nimage).*weights(:,n));
        %output is the activation of the output node. a = sig(Z).
        %sigActivation is a non-linear function that constrains the sum to 
        %be inside a value between 0 and 1.
        output(n) = sigActivation(sumNode);
        %The difference between the actual output and the target output is
        %found.
        error(n) = target_output(n)-output(n)
        %The weights are updated with respect to the error of the output,
        %the magnitude of the input and the learning rate. 
        %w = w + Input*error*LR
        weights(:,n) = weights(:,n) + (images(:,nimage)*error(n)*LR); 
    end
    
    fprintf("Epoch: %d\n", nimage);
    %Was previously printing these variables to view the progress of the
    %system as it trained.
    target_output;
    output;
    error;
    
end
%%
%%This section is to evaluate the model. The number of testing images we
%%want to attempt is set by numTests.
numTests = 100;

%Metrics to measure the accuracy at the end of testing
correctPredictions = 0;
incorrectPredictions = 0;

for nimage = 1:numTests
    for n = 1:numOutputs
        %This code is borrowed from the training, but without the error
        %calculation and weight updating
        sumNode = 0;
        sumNode = sum(test_images(:,nimage).*weights(:,n));
        %output(n) = sumNode/numInputs;  %Original activation function
        output(n) = sigActivation(sumNode);
    end

    %Calculating the most activated node, this will correspond to the
    %predicted number
    [val id] = max(output);
    
    %Printing predictions and actual value of testing image
    if id == 1
        fprintf("The NN predicts the number is %d\n", Num1)
    end
    if id == 2
        fprintf("The NN predicts the number is %d\n", Num2)
    end
    
    fprintf("The actual number is %i \n", testing_labels_selected(nimage))
    
    %Updating the statistics on the prediction and if the prediction was
    %false it displays that image.
    if NumArray(id) == testing_labels_selected(nimage)
        correctPredictions = correctPredictions + 1;
    else
        incorrectPredictions = incorrectPredictions + 1;
        figure;
        imshow(testing_images_selected(:,:,nimage));
        title(['Predicted ', num2str(NumArray(id)), ' but Actual is ', num2str(testing_labels_selected(nimage))]);
    end
    
end

%Printing final accuracy
accuracy = (correctPredictions)/(correctPredictions+incorrectPredictions)
            
    
    

