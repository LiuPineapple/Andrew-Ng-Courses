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
Y=X*theta;Y=Y-y;
J=Y'*Y/(2*m);
J=J+(theta'*theta-theta(1,1)^2)*lambda/(2*m);
grad=(Y'*X)'/m;
for i=2:length(grad)
    grad(i)=grad(i)+lambda*theta(i)/m;
end



% =========================================================================

grad = grad(:);

end
