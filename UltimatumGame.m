function [observation, strategy, oldStra] = UltimatumGame(adj,s,oldStra,noise)
%%
% adj -> adjancency matrix of network
% s   -> observation rate/times
% oldStra -> old strategies

%% data preparation
[m,n] = size(adj);
T = s;
D = adj;
p=zeros(T,n);   % strategy for proposer
q=zeros(T,n);   % strategy for responsor
% u=zeros(n,n);
U=zeros(T,n);   % real payoff gain 
Iu=zeros(T,n,n);% visual paypff gain

%% gaming stage
if isempty(oldStra)
    p(1,:)=rand(1,n);
    q(1,:)=rand(1,n);
else
    p(1,:) = oldStra(:,1);
    q(1,:) = oldStra(:,2);
end

% first gaming
t=1;
for i=1:n
 for j=1:n
    if i==j
        Iu(t,j,i)=0;
    else
    if p(t,i)>=q(t,j)
        if p(t,j)>=q(t,i)
            Iu(t,j,i)=1-p(t,i)+p(t,j);
        else
            Iu(t,j,i)=1-p(t,i);
        end
    else
       if p(t,j)>=q(t,i)
            Iu(t,j,i)=p(t,j);
        else
            Iu(t,j,i)=0;
       end
    end
   end
 end
 U(t,i)=Iu(t,:,i)*D(:,i);
end

% gaming next
for t=2:T
    tempD=D;
    tempD=tempD+rand(n,n);
    for i=1:n
        % learning stage
        [temp1,temp2]=sort(tempD(i,:),'descend');
        if U(t-1,i)<U(t-1,temp2(1,1))
            W=(U(t-1,temp2(1,1))-U(t-1,i))/(2*max(sum(D(i,:)),sum(D(temp2(1,1),:))));
            P=rand(1);
            if W>P
                delta1=0.1*(rand(1)-0.5);
                p(t,i)=p(t-1,temp2(1,1))+delta1;
                delta2=0.1*(rand(1)-0.5);
                q(t,i)=q(t-1,temp2(1,1))+delta2;
                if p(t,i)<0
                    p(t,i)=0;
                end
                if q(t,i)<0
                    q(t,i)=0;
                end
                if p(t,i)>1
                    p(t,i)=1;
                end
                if q(t,i)>1
                    q(t,i)=1;
                end
            else
                delta1=0.1*(rand(1)-0.5);
                p(t,i)=p(t-1,i)+delta1;
                delta2=0.1*(rand(1)-0.5);
                q(t,i)=q(t-1,i)+delta2;
                if p(t,i)<0
                    p(t,i)=0;
                end
                if q(t,i)<0
                    q(t,i)=0;
                end
                if p(t,i)>1
                    p(t,i)=1;
                end
                if q(t,i)>1
                    q(t,i)=1;
                end
            end
        else
            % random choose strategy
            delta1=0.1*(rand(1)-0.5);
            p(t,i)=p(t-1,i)+delta1;
            delta2=0.1*(rand(1)-0.5);
            q(t,i)=q(t-1,i)+delta2;
            if p(t,i)<0
                p(t,i)=0;
            end
            if q(t,i)<0
                q(t,i)=0;
            end
            if p(t,i)>1
                p(t,i)=1;
            end
            if q(t,i)>1
                q(t,i)=1;
            end
        end
    end
    % game stage
    for i=1:n
        for j=1:n
        if i==j
            Iu(t,i,j)=0;
        else
        if p(t,i)>=q(t,j)
            if p(t,j)>=q(t,i)
                Iu(t,j,i)=1-p(t,i)+p(t,j);
            else
                Iu(t,j,i)=1-p(t,i);
            end
        else
        if p(t,j)>=q(t,i)
                Iu(t,j,i)=p(t,j);
            else
                Iu(t,j,i)=0;
        end
        end
        end
        end
        U(t,i)=Iu(t,:,i)*D(:,i);
    end
end

if (noise > 0 )
    e=randn(T,n);
    sigma=noise.^2;
    U=U+sigma*e;
end

observation = U;
strategy(:,:,1) = p;
strategy(:,:,2) = q;

oldStra(:,1) = p(T,:);
oldStra(:,2) = q(T,:);

for i = 1:n
    tempIu(:,:,i)=Iu(:,:,i);
end

for i=1:n
    A=tempIu(:,:,i); 
    y=U(:,i);
    observation(:,i) = y;
    strategy(:,:,i) = A;
end

end

