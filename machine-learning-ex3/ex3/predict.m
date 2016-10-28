function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% add the column of 1's to X:
X = [ones(m, 1) X];
a1 = X; % for better generalization of code.

%prediction at layer 2:
z2 = a1 * Theta1';

%compute a2:
a2 = sigmoid(z2);

%add a column of 1s of *size k* to a2:
k = size(a2,1);
a2 = [ones(k, 1) a2];

% See a pitfall below.
% note that adding a column of 1s BEFORE 'sigmoiding' doesn't work.
% b/c sigmoid(1) = 0.7311 but we need a column of 1s. Earlier program
% was right as far as dimensions go but didn't work bc of this glitch.
% Did 

%compute z3:
z3 = a2*Theta2';

%compute a3, the prediction matrix at layer 3;
a3 = sigmoid(z3);

%set p to be the index of the max of each row is p
[Max_p,p] = max(a3, [], 2);









% =========================================================================


end
