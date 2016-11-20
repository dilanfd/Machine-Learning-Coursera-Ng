function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

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

%initialize a matrix to hold the erros and the corresponding sigmas and Cs
results_matrix = [];

% use two nested for loops to iterate over every possible valuees for 
% C and sigma and store them in temp variables C_temp and sigma_temp
for C_temp = [0.01 0.03 0.1 0.3 1 3 10 30]
    for sigma_temp = [0.01 0.03 0.1 0.3 1 3 10 30]
        model = svmTrain(X, y, C_temp,@(x1,x2) gaussianKernel(x1,x2,sigma_temp));
        predictions = svmPredict(model, Xval);
        pred_error = mean(double(predictions ~= yval));
        %create the results matrix: for each iteration of the outer
        % loop and each iteration of the innner loop
        % write a row that has pred_error, C_temp, sigma_temp
        % results_matrix is 64 by 3. First column is pred_error, second
        % column is C_temp and third is sigma_temp.
        results_matrix = [results_matrix; pred_error, C_temp, sigma_temp];
    end
end
%store the min_error and most importantly the index (i.e.,in which row does the
% min of first column appear?) of the min_error...
[min_error, index_of_min_error] = min(results_matrix(:,1));

%use the obove stored index to set your C value and sigma value
C = results_matrix(index_of_min_error, 2);  % bc second row of result_matrix
                                            % C_temp
sigma = results_matrix(index_of_min_error,3); % 3rd row of results matrix is
                                              % sigma_temp





% =========================================================================

end
