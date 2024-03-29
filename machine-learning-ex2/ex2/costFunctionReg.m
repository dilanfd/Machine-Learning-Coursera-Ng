function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%cost function

J = (1/m)*sum(-y'*log(sigmoid(X*theta))-(1-y')*log(1-sigmoid(X*theta))) + (lambda/(2*m))*dot(theta,theta) - (lambda/(2*m))*(theta(1))^2


% grad function. We create a temporary variable to hold on to the
% vectorized computation in (X' * (sigmoid(X * theta) - y)). Since we need
% the element wise values. We can't do a straight up vectorization because
% the gradient is calculuated differently for theta(1). For theta(i) with i
% >= 2 we have the regularized gradient. 

temp  = (X' * (sigmoid(X * theta) - y)); % stored it in a temp variable.
%compute \partial j over \partial theta_0
grad(1) = (1/m)* temp(1);

% compute \partial j over \partial theta_i for i >= 2
for i = 2:length(theta)
    grad(i) = (1/m)*temp(i) +(lambda/m)*theta(i);
end



% =============================================================

end
