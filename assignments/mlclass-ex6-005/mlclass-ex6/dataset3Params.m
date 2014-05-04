function [C, sigma, error_master] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 

% Outline for implementation
% 1) loop on test C values
% C_vect = [0.01; 1; 3];
% sigma_vect = [0.01; 1; 3];
C_vect = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigma_vect = [1];

C_vect = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigma_vect = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

depth = 2;
error_master = [];

%for C = 0.01 * 10.^[0:depth]
for i = 1:size(C_vect)(1)
	C = C_vect(i);

	% 2) nest loop on test Sigma values
	%for sigma = 0.03 * 10.^[0:depth]
	for j = 1:size(sigma_vect)(1)
		sigma = sigma_vect(j);
		
		% 3) calculate test model theta values based on C and Sigma inputs on 
		%    training set data; as well as the differential between 
		% two points x1, x2. pass in your test C and Sigma values.
		x1 = X(:,1); 
		x2 = X(:,2);
		model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
		%model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);


	    %  4) call svmPredict(...) on the cross-validation dataset; 
		% and using the test model theta from (3) to calculate test predictions
		predictions = svmPredict(model, Xval);

	    % 5) look at the ex6.pdf to find out how to calculate the error between
	    % the test predictions from (4) to calculate test error
	    classificationError = mean(double(predictions ~= yval));

	    %  6) test if test error is the smallest seen so far, with attention to setting
	     %     it for the first time.  If so, save the C and sigma values as the best 
	      %    values found so far
	    if (exist("minError", "var") == 1)
	    	if( classificationError < minError)
	    		minError = classificationError;
	    		bestC = C;
	    		bestSigma = sigma;
	    	endif
	    else
	    	minError = classificationError;
	    	bestC = C;
	    	bestSigma = sigma;
	    endif


		# append element to debug matrix
		error_master(end+1,:) = [C, sigma, classificationError];  

	endfor
endfor

% return values
C = bestC;
sigma = bestSigma;

% =========================================================================

end
