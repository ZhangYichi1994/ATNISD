function [theta] = identification(observation, strategy, lambda)
if nargin < 3
    lambda = [1, 1];
end
% observation 结构 （x, i , t）
% strategy 结构 (x, y, i, t)
evolveTime = size(observation,3); % how many time
p = size(observation,2);          % node number
theta = zeros(p,p,evolveTime);


for i = 1:p         % p is node number
    %% 整理出每个点的观测值 
    observeTemp = zeros(size(observation,1),evolveTime);
    straTemp = zeros(size(strategy,1),size(strategy,2),evolveTime);
    for t = 1:evolveTime
        observeTemp(:,t) = observation(:,i,t);
        straTemp(:,:,t) = strategy(:,:,i,t);
    end
    
    % fist method
    cvx_begin
        variable x(p,evolveTime)
        minimize( sigmaAdd(observeTemp, straTemp, x, lambda) );
    cvx_end
    
    for t = 1:evolveTime
        theta(:,i,t) = x(:,t);
    end
    
end

end

function addition = sigmaAdd(observe,strategy,x, lambda)
t = size(observe,2);
a = 0;
for i = 1:t
    a = a + norm(observe(:,i) - strategy(:,:,i) * x(:,i),2) + lambda(1) * norm(x(:,i),1);
end
for i = 2:t
    a = a + lambda(2) * norm(x(:,i) - x (:,i-1),1);
end 
addition = a;
end

function A = tvMatrix(x)
    [m,n] = size(x);
    A = zeros(m-1,n);
    for i = 1:m-1
        A(i,i) = 1;
        A(i,i+1) = -1
    end
end