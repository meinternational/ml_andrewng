function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);
X = [ones(m, 1) X];
% forward propagation
z1 = X*Theta1';
a1 = sigmoid(z1);
a1 = [ones(m, 1) a1];
z2 = a1*Theta2';
a2 = sigmoid(z2);

p = a2./repmat(sum(a2,2),[1 num_labels]);
[~,p] = max(p,[],2);






% =========================================================================


end
