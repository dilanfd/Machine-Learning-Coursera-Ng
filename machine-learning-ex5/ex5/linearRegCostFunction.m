function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%create a temp variable with the temporary error.
%used for better code readability.
tempError = X*theta - y;

%temp thetas first term is set to zero and then
% it uses that to vectorize our code. Nice little
% device
temp_theta = theta;
temp_theta(1) = 0;

J = (1/(2*m))*(tempError'*tempError) + ...
    (lambda/(2*m)) *sum(temp_theta.^2);

                                           
%computing the gradient....

% unregularized gradient
grad = (1/m)* (X'*tempError);

%update grad for regularization as follows
grad = grad + (lambda/m)*temp_theta;









% =========================================================================

grad = grad(:);

end
