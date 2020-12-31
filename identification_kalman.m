function [theta] = identification_kalman(observation, strategy, lambda)
if nargin < 3
    lambda = 4;
end
% observation 结构 （x, i , t）
% strategy 结构 (x, y, i, t)
evolveTime = size(observation,3); % how many time
p = size(observation,2);          % node number
theta = zeros(p,p,evolveTime);
measureCount = size(observation,1)


for i = 1:p         % p is node number
    %% 整理出每个点的观测值 
    observeTemp = zeros(size(observation,1),evolveTime);
    straTemp = zeros(size(strategy,1),size(strategy,2),evolveTime);
    for t = 1:evolveTime
        observeTemp(:,t) = observation(:,i,t);
        straTemp(:,:,t) = strategy(:,:,i,t);
    end
    
    % kalman filter 
    Sav =  16 ; %8 16 25; % round( n/log2(m/n) ) %S1 = S - 2;
    cc = 1; %10 %multiply 'sigma' and 'a' by 'cc'
    sigma0 =  cc* (1/3)*sqrt(Sav/measureCount);
    sigsys = 1;
    siginit = 3;
    R = (sigma0^2)*eye(measureCount);
    Q = (sigsys^2)*eye(p); %F = eye(p);
    Pi0 = (siginit^2)*eye(p);
    [x_upd,T_hat] = dynamicalKalman(observeTemp, straTemp, Pi0, Q, R, lambda);
%     cvx_begin
%         variable x(p,evolveTime)
%         minimize( sigmaAdd(observeTemp, straTemp, x, lambda) );
%     cvx_end
    
    
    for t = 1:evolveTime
        theta(:,i,t) = x_upd(:,t);
    end
    
end

end