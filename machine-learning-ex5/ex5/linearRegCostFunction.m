function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
m = length(y);
J = (X*theta-y).^2;
J = (0.5/m)*sum(J(:));
% don't regularize theta 1 (the constant)
J = J + 0.5*(lambda/m)*sum(theta(2:end).^2);

lambda = lambda*ones(length(theta),1);
lambda(1) = 0;
grad = (1/m)*(((X*theta)-y)'*X) +  ((1/m)*lambda.*theta)';
grad = grad(:);

end
