%%Author: Philip Giles, UCD
%
%%THIS CODE IS NOT FUNCTIONING CORRECTLY YET. 
%
%%This script builds upon the other other scripts and adds a hidden layer
%%between the inputs and outputs. Instead of just weights, this network
%%also has a bias for every node. The backpropogation equations used were
%%found mostly on http://neuralnetworksanddeeplearning.com/chap2.html . 

numOutputs = 10;
sizeHiddenLayer = 16;
%Num1 = 3;
%Num2 = 7;  
NumArray = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
LR = 0.02; %Learning Rate

[training_images, training_labels] = readMNIST('MNIST_TRAINING_IMAGES.idx3-ubyte', 'MNIST_TRAINING_LABELS.idx1-ubyte', 60000, 0);
[testing_images, testing_labels]   = readMNIST('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte', 10000, 0);

test_images = reshape(testing_images, size(testing_images, 1)*size(testing_images, 2), size(testing_images, 3));
images = reshape(training_images, size(training_images, 1)*size(training_images, 2), size(training_images, 3));
%%

numInputs = size(images,1);
weightsLayer1 = rand(sizeHiddenLayer, numInputs);
biasLayer1 = rand(1, sizeHiddenLayer);
weightsOutput = rand(numOutputs, sizeHiddenLayer);
biasOutput = rand(1, numOutputs);
layer1Activations = zeros(1,sizeHiddenLayer);
errorOutput = zeros(1, numOutputs);
errorHiddenLayer = zeros(1, sizeHiddenLayer);
output = zeros(1, numOutputs);
target_output = zeros(1,numOutputs);
ZHiddenLayer = zeros(1, sizeHiddenLayer);
ZOutput = zeros(1, numOutputs);
sigDerZHiddenLayer = zeros(1, sizeHiddenLayer);   %This vector stores the derivative of the sigmoid of Z for hidden layer
sigDerZOutput = zeros(1, numOutputs);    %This vector stores the derivative of the sigmoid of Z for output layer
Cost =  zeros(1,numOutputs);
numSelected = size(training_labels, 1);

for nimage = 1:1:numSelected
    
    target_output = zeros(1,numOutputs);
    target_output(training_labels(nimage)+1) = 1;

    %Calculate the activations in the first layer
    
    for n = 1:sizeHiddenLayer
        sumNode = 0;
        sumNode = sum(images(:,nimage).*weightsLayer1(n,:).');
        sumNode = sumNode + biasLayer1(:,n);
        ZHiddenLayer(n) = sumNode;
        layer1Activations(:,n) = sigActivation(sumNode);
        sigDerZHiddenLayer(n) = layer1Activations(:,n)*(1-layer1Activations(:,n)); %saving this value for later
    end
    
    
    %this is for output node 1 (number 2)
    for n = 1:1:numOutputs
        sumNode = 0;
        sumNode = sum(layer1Activations(:).*weightsOutput(n,:).');
        %output(n) = sumNode/numInputs;
        sumNode = sumNode + biasOutput(:,n);
        ZOutput(1,n) = sumNode;
        output(n) = sigActivation(sumNode);
        sigDerZOutput(n) = output(:,n)*(1-output(:,n));
    end
    
    %%update cost
    Cost = ((output - target_output).^2);
    CostMag = sum(Cost)*0.5;
    
    errorOutput = (output-target_output).*sigDerZOutput;
    weightsOutput = weightsOutput - LR*(errorOutput.'*layer1Activations);
    biasOutput = biasOutput - errorOutput;
    
    errorHiddenLayer = ((errorOutput)*(weightsOutput).*sigDerZHiddenLayer);
    weightsLayer1 = weightsLayer1 - LR*(images(:,nimage)*errorHiddenLayer).';
    biasLayer1 = biasLayer1 - errorHiddenLayer;
    
    
    
    fprintf("Epoch: %d\n", nimage);
    fprintf("Cost: %d\n", CostMag);
    target_output;
    output;
    
end
%%
numTests = 100;

correctPredictions = 0;
incorrectPredictions = 0;

for nimage = 1:1:numTests
    for n = 1:sizeHiddenLayer
        sumNode = 0;
        sumNode = sum(images(:,nimage).*weightsLayer1(n,:).');
        sumNode = sumNode + biasLayer1(:,n);
        layer1Activations(:,n) = sigActivation(sumNode);
    end
    

    for n = 1:1:numOutputs
        sumNode = 0;
        sumNode = sum(layer1Activations(:).*weightsOutput(n,:).');
        %output(n) = sumNode/numInputs;
        sumNode = sumNode + biasOutput(:,n);
        output(n) = sigActivation(sumNode);
    end
    sumNode
    [val id] = max(output);
    
    fprintf("Predicted number is %i \n", NumArray(id))
    fprintf("The actual number is %i \n", testing_labels(nimage))
    
    if NumArray(id) == testing_labels(nimage)
        correctPredictions = correctPredictions + 1;
    else
        incorrectPredictions = incorrectPredictions + 1;
%         figure;
%         imshow(testing_images(:,:,nimage));
%         title(['Predicted ', num2str(NumArray(id)), ' but Actual is ', num2str(testing_labels(nimage))]);
    end
    
end

accuracy = (correctPredictions)/(correctPredictions+incorrectPredictions)

