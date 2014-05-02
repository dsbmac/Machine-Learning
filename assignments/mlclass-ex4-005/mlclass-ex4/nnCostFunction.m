function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% local directory for octave
% cd 'C:/Users/dsbmac/Documents/Professional Development/Machine Learning/assignments/mlclass-ex4-005/mlclass-ex4'

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% a1 equals the X input matrix with a column of 1's added (bias units)

a1 = [ones(m,1), X];

 % z2 equals the product of a1 and Theta1
z2 = 	a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a2)(1,:),1), a2];

% a3 is product of a2 with trans(Theta2)
z3 = a2 * Theta2';
a3 = sigmoid(z3);

[prediction, p] = max(a3, [], 2);

y_matrix = eye(num_labels)(y,:);


diff1 = -y_matrix .* log(a3);
diff2 = (1 .- y_matrix) .* (log(1 .- a3));

diff = diff1 - diff2;
J = sum(1/m * sum(diff));


% regularized cost function
% compute the regularization terms separately

% ignore the bias unit
Theta1Rest = Theta1(:, 2:end);
Theta2Rest = Theta2(:, 2:end);

regularization1 =  (lambda / (2*m)) * sum( sum(Theta1Rest .^ 2) );
regularization2 =  (lambda / (2*m)) * sum( sum(Theta2Rest .^ 2) );


% then add them to the unregularized cost from Step 3.
J = J + regularization1	+ regularization2;


% Backpropagation

% calculate d3
d3 = a3 .- y_matrix;

% calculate d2
d2 = (d3 * Theta2Rest) .* ( sigmoidGradient(z2) ) ;

% calculate Delta2
Delta2 = d3' * a2;
Delta1 = d2' * a1;

% calculate non-regularized theta gradients
Theta1_grad = (1/m) * Delta1;
Theta2_grad = (1/m) * Delta2;


% regularized theta gradients

% calculate regularization terms
Theta1_gradReg = (lambda / m) * Theta1;
Theta2_gradReg = (lambda / m) * Theta2;

% overwrite the regularizatio bias units to 0
Theta1_gradReg(:,1) = 0;
Theta2_gradReg(:,1) = 0;

Theta1_grad = Theta1_grad + Theta1_gradReg;
Theta2_grad = Theta2_grad + Theta2_gradReg;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
