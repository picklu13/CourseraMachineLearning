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

cost = 0;
reg_sum = 0;
for i = 1:m

	x_i = X(i,:);
	y_i = y(i);
	h_theta = sigmoid(x_i * theta);
	grad_factor = h_theta - y_i;
	cost =  cost + -y_i * log(h_theta)  - (1-y_i) * log(1 - h_theta);

	for k = 1:size(grad)
		grad(k) = grad(k) + (grad_factor *  x_i(k));
	end
	


end


for j = 2:size(theta)
	reg_sum = reg_sum + theta(j) * theta(j);
end

J = (1 / m ) * cost + (lambda / (2*m) ) * reg_sum;

grad(1) = grad(1) / m;
for k  = 2:size(grad)
	grad(k) = grad(k) / m + (lambda /m ) * theta(k) ;
end







% =============================================================

end
