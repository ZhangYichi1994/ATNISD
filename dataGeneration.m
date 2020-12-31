function [observation, strategy, Adjset, straSeries, sampleNumMat] = dataGeneration(parameter)
p = parameter.networkSize;        % nodes number of network
d = parameter.averageDegree;      % averaeg degree in first network
r = parameter.changeRate;         % how many edges will change in one cycle
s = parameter.sampleRate;         % in one epoch, we need how many data
sampleModel = parameter.sampleModel; % sampleing model 
weightModel = parameter.weightModel; % 
times = parameter.times;     % evolving times == how many times do network vary
mode = parameter.mode;
networkMode = parameter.networkMode;
noise = parameter.noise;
% interval = epochInterval;   % how many epoches in one unit varying time 
%% initial network
adj = initialNetwork(p,d,networkMode, weightModel);
adjVec = reshape(adj,p*p,1);
adjWeight = ones(p*p,1);
oldStra = [];
sampleNumMat = zeros(1,times);

observation = zeros(6, p, times);
strategy = zeros(6, p, p, times);

for evolve = 1:times
%         Adjset(:,:,evolve) = adj - diag(sum(adj));
    Adjset(:,:,evolve) = adj;
%% netowrk sampling
    [voltage, current, oldStra, straStatic, sampleNum] = sampling(adj,s, mode, oldStra, noise, sampleModel);   
    observation(1:sampleNum,:,evolve) = current;
    strategy(1:sampleNum,:,:,evolve) = voltage;
    sampleNumMat(1, times) = sampleNum;
    
    coperationRate = sum(straStatic)./size(straStatic,1);             % statistics on the cooperation proportion  
    straSeries(:,[3*evolve-2:3*evolve]) = coperationRate;
    
    adjTemp = adj;
%% network evolving
    % 定制一个权重，每次加减边的时候权重变化（+1？？）
    % 根据权重进行采样, 选择要变化的边界
    for count = 1:r
        % 减去边
        population = find(adjVec ~= 0);
        select = randsample(population,1,true,1./adjWeight(population));
        locaY = ceil(select ./ p);  % 取整
        locaX = rem(select,p);      % 取余
        select2 = locaX * p + locaY ;
        if locaX == 0
            locaX = p;
        end
        for counti = 1:p
            for countj = 1:p
                if (adj(counti,countj) ~= 0)
                    adjTemp(counti,countj) = 1;
                end
            end
        end     
        sumY = sum(adjTemp(:,locaY));
        sumX = sum(adjTemp(:,locaX));
        while( (sumY <= 1) || (sumX <= 1) )
            select = randsample(population,1,true,1./adjWeight(population));
            locaY = ceil(select ./ p);  % 取整
            locaX = rem(select,p);      % 取余
            select2 = locaX * p + locaY ;
            if locaX == 0
                locaX = p;
            end
            sumY = sum(adjTemp(:,locaY));
            sumX = sum(adjTemp(:,locaX));
        end
        adj(locaX,locaY) = 0; %删掉一个边
        adj(locaY,locaX) = 0; %对称位置删掉一条边
        adjVec = reshape(adj,p*p,1);
        adjWeight(select,1)  = adjWeight(select,1) + 1;% 该边的权重加一(操作数加一)
        adjWeight(select2,1) = adjWeight(select2,1) + 1;%对称位置权重加一

        % 添加边
        population = find(adjVec == 0);
        select = randsample(population,1,true,1./adjWeight(population));
        locaY = ceil(select ./ p);
        locaX = rem(select,p);
        select2 = locaX * p + locaY;
        if locaX == 0
            locaX = p;
        end
        edgeWeight = randWeight(weightModel);
        adj(locaX,locaY) = edgeWeight;% 增加一条边界
        adj(locaY,locaX) = edgeWeight;% 对称位置增加一条边
        adjVec = reshape(adj,p*p,1);
        adjWeight(select,1) = adjWeight(select,1) + 1;% 该边权重加一
        adjWeight(select2,1) = adjWeight(select2,1) + 1;%对称位置权重加一
    end
    

end
end

%% initial network
function adj = initialNetwork(p,d, mode, weightModel)      
% p -> number of nodes
% d -> average degree
% mode -> 1:PNAS method , 2: BA network
if mode ==1 
    adj = zeros(p,p);
    degree = zeros(p,1);
    for i = 1:p
        for j = 1:max(0,d-degree(i))
            flag = 0;
            while(flag == 0)
                select =  unidrnd(p);
                if (adj(select,i) == 0 && (select ~= i))
                   adj(select,i) = 1 * randWeight(weightModel);    degree(i) = degree(i) + 1;
                   %adj(select,i) = 1;    degree(i) = degree(i) + 1;
                   adj(i,select) = adj(select, i);      degree(select) = degree(select) + 1;
                   flag = 1;
                end
            end
        end 
    end
end
if mode ==2
    adj = SFNG(p,d,2*d, weightModel);
end

end

%% sample for one time epoch with different observation
function [voltage, current, oldStra, Stra, sampleTime] = sampling(adj,s, mode, oldStra, noise, sampleModel)
Stra = [];
SIZE = size(adj,1);
voltage = zeros(s,SIZE,SIZE);
current = zeros(s,SIZE);

if mode == 1    % 通牒博弈
    if sampleModel == 1
        sampleTime = s + int8(2 * rand(1));
    else
        sampleTime = s;
    end
    [observation, strategy, oldStra] = UltimatumGame(adj,sampleTime,oldStra,noise);
    current = observation;
    voltage = strategy;
end
if mode == 2   % 洛伦兹动力学
    if sampleModel == 1
        sampleTime = s + int8(2 * rand(1));
    else
        sampleTime = s;
    end
    [observation, state] = LorentzModel(adj, sampleTime, oldStra, noise);
    current = observation;
    voltage = state;
    sampleTime = (sampleTime-1) * 3;
end

end

%% generate random number between [-1,-0.5]U[0.5,1]
function a = randWeight(weightModel)
    a = rand;
    if a < 0.5
        a = a - 1;
    end
    if weightModel == 0
        a = 1;
    end
end

%% generate BA scale-free network
function SFNet = SFNG(Nodes, mlinks, seed, weightModel)
seed = full(seed);
pos = length(seed);
%if (Nodes < pos) || (mlinks > pos) || (pos < 1) || (size(size(seed)) ~= 2) || (mlinks < 1) || (seed ~= seed') || (sum(diag(seed)) ~= 0)
%    error('invalid parameter value(s)');
%end
%if mlinks > 5 || Nodes > 15000 || pos > 15000
%    warning('Abnormally large value(s) may cause long processing time');
%end
rand('state',sum(100*clock));
Net = zeros(Nodes, Nodes, 'single');
Net(1:pos,1:pos) = seed;
sumlinks = sum(sum(Net));
while pos < Nodes
    pos = pos + 1;
    linkage = 0;
    while linkage ~= mlinks
        rnode = ceil(rand * pos);
        deg = sum(Net(:,rnode)) * 2;
        rlink = rand * 1;
        if rlink < deg / sumlinks && Net(pos,rnode) == 0 && Net(rnode,pos) == 0
            edgeWeight = 1 * randWeight(weightModel);
            Net(pos,rnode) = edgeWeight;
            Net(rnode,pos) = edgeWeight;
            linkage = linkage + 1;
            sumlinks = sumlinks + 2;
        end
        tempMatrix = Net(1:pos,1:pos);
        tempMatrix(logical(eye(size(tempMatrix))))=1;
        if ( min(tempMatrix) ~= 0)
            break;
        end
    end
end
for i = 1:Nodes
    Net(i,i) = 0;
end
clear Nodes deg linkage pos rlink rnode sumlinks mlinks
SFNet = Net;
end