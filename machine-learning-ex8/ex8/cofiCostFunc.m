function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold X and Theta matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%
error = (X*Theta' - Y);
% consider only those values of error which correspond to 1's in R.
error_factor = error.* R; 
% elementwise multiplication by R gives you the desired outcome.
% notice how we can't use
% error_fac = error(R ==1) bc the latter becomes a column matrix and
% therefore not useful in computing X_grad and Theta_grad in vectorized
% implementations. This is hard to see because it calculates J correctly
% because of the way we have implemented it. The following is a nice trick
% to do that without doing either an explicit sum or god forbid a for -
% loop. :)
J = (1/2)* (error_factor(:)' * error_factor(:)); 
% above complete vectorized implementation. 

 
X_grad = error_factor * Theta;
Theta_grad = error_factor' * X;

% regularization
tempTheta = Theta(:); 
% make long vecotors out of Theta and X's and then use that to avoid for
% loops or double sums. Implemmentation is completely vectorized.
tempX = X(:);
J = J + ((lambda/2)*(tempTheta'*tempTheta)) + ...
    ((lambda/2)* (tempX'*tempX));

X_grad = X_grad + lambda * X;
Theta_grad = Theta_grad + lambda *Theta;












% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
