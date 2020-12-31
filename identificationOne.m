function [theta] = identificationOne(observation, strategy, lambda)
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
    
    xResult = zeros(p,evolveTime);
    for t = 1:evolveTime
        Yt = observeTemp(:,t);
        At = straTemp(:,:,t);
        cvx_begin
            variable x(p)
            minimize( 0.5*norm(Yt-At*x,2) + lambda(1) * norm(x,1) );
        cvx_end
        xResult(:,t) = x;
    end

    for t = 1:evolveTime
        theta(:,i,:) = xResult(:,:);
    end
    
end
% diagonal equal to 0
for i = 1:evolveTime
    temp = theta(:,:,i);
    temp(logical(eye(size(temp))))=0;
    theta(:,:,i)=temp;
end
end