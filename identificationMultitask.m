function [theta] = identificationMultitask(observation, strategy)
% observation 结构 （x, i , t）
% strategy 结构 (x, y, i, t)
evolveTime = size(observation,3); % how many time
p = size(observation,2);          % node number
observeNum = size(observation,1);
theta = zeros(p,p,evolveTime);

for i = 2:p         % p is node number
    %% 整理出每个点的观测值 
    
    for t = 1:evolveTime
        observeTemp{t} = observation(:,i,t);
        tempStr = strategy(:,:,i,t);    %  去掉第i列，因为全为零
        tempStr(:,i) = [];
        straTemp{t} = tempStr;
    end
    
    a = 1e2/0.1; b = 1;
    weights = mt_CS(straTemp,observeTemp, a,b,1e-8);
    
    fprintf('Loop number is: %d',i);
    for t = 1:evolveTime
        temp = weights(:,t);
        temp2 = zeros(p,1);
        temp2(1:i-1) = temp(1:i-1);
        temp2(i) = 0;
        temp2(i+1:end) = temp(i:end);
        theta(:,i,t) = temp2;
    end

end

end