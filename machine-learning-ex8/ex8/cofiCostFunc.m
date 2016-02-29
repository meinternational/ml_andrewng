function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
                              
% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));
[m,n] = size(R);
o=size(X,2);

% gradient
J = (X*Theta'-Y).^2;
J = sum(sum(J(R==1)))/2;
sum_dummy=0;
for j=1:n
    for k=1:o
        sum_dummy = sum_dummy + Theta(j,k)^2;     
    end
end
J = J+0.5*lambda*sum_dummy;
sum_dummy=0;
for i=1:m
    for k=1:o
        sum_dummy = sum_dummy + X(i,k)^2;     
    end
end
J = J+0.5*lambda*sum_dummy;


for i=1:m
    idx = find(R(i, :)==1);
    Thetatemp = Theta(idx, :);
    Ytemp = Y(i, idx);
    X_grad(i, :) = (X(i, :)*Thetatemp' - Ytemp) * Thetatemp + lambda*(X(i, :));
end

for j=1:n
    for k=1:o
        sum_dummy = 0;
        for i=1:m
            if R(i,j)==1
                sum_dummy=sum_dummy+(X(i,:)*Theta(j,:)'-Y(i,j))*X(i,k);
            end
        end
        Theta_grad(j,k)=sum_dummy;
    end
    Theta_grad(j,:) = Theta_grad(j,:) + lambda*Theta(j,:); 
end



grad = [X_grad(:); Theta_grad(:)];

