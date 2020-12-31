function [observation, state, oldState]=LorentzModel(adj, s, oldState, noise)
%%
% adj -> adjancency matrix of network
% s   -> observation rate/times
% oldState -> old state

%% data preparation
[m,n] = size(adj);
x=zeros(1,n);   % state x axis
y=zeros(1,n);   % state y axis
z=zeros(1,n);   % state z axis

num_nodes = n;
h = 0.0000001;
num_meas =64;%45              %观测点数(<34)64,196
d_dimension = 3;              %每个振子的维数

iterationTime =s;        %迭代次数

history=[];                   %记录轨迹上所有点

inner_mat=[0,1,0;0,0,1;1,0,0];%内联矩阵

outer_mat = adj - diag(sum(adj));
% n = d_dimension*(num_nodes+1)+1;%方法一中向量的维数




%% running stage
if isempty(oldState)
    Point_Init = random('Normal',0,100,3,num_nodes);
else
    Point_Init = oldState;
end



Network_parameters.Point_Init = Point_Init;
Network_parameters.iterationTime = iterationTime;
Network_parameters.N = num_nodes;
Network_parameters.AA = inner_mat;
Network_parameters.C = outer_mat;
Network_parameters.h = h;
[history] = Net_Generating(Network_parameters);

oldState = history(:,:,iterationTime);

observation = zeros(3 * (iterationTime-1), num_nodes);
state = zeros(3*(iterationTime-1), num_nodes, num_nodes);
for i = 1:num_nodes             % 第i个点开始
    % observation
    temp2 = zeros(3 , iterationTime-1);
    temp3 = zeros(3 , iterationTime-1);
    for j = 1:(iterationTime - 1)
        temp = (history(:,i,j+1) - history(:,i,j)) / h;
        x = history(1,i,j);
        y = history(2,i,j);
        z = history(3,i,j);
        u=10*(y-x);
        v=28*x-y-x*z;
        w=x*y-8/3*z;
        Q=[u,v,w]';
        temp2(:,j) = temp - Q;
        temp3(:,j) = history2(:,i,j);
    end
    
    observation(:,i) = reshape(temp2, 3*(iterationTime-1),1);
    if noise ~= 0
        e = rand(3*(iterationTime-1),1);
        sigma=noise.^2;
        observation(:,i) = observation(:, i) + sigma*e;
    end
    
    % state
    phi2 = zeros(3*(iterationTime-1), num_nodes);
    for j = 1:(iterationTime - 1)
        phi1 = zeros(3,num_nodes);
%             phi1(:,k) = inner_mat * (history(:,i,j) - history(:,k,j));    % j时间， i点与k点的差
        phi1(:,:) = inner_mat * ( history(:,:,j));    % j时间， i点与k点的差
        phi2(1+3*(j-1):3*j,:) = phi1;
    end
    state(:,:,i) = phi2;
    
    
end



end