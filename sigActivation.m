%%Author: Philip Giles, UCD
function y = sigActivation(x)
%sigActivation: Activation function for my "neural network". Constrains the
%input value x between 0 and 1.
y = (1/(1+exp(-x)));
end

