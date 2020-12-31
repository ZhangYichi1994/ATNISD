%KF-CSres-LS (no final KF step) with deletion step
function [x_upd,T_hat] = dynamicalKalman(y, A, Pi0, Q, R, lambdap)
if nargin < 6
    lambdap = 4;
end
[n,m,tot] = size(A);
% global S 
% global Smax

% T_hat_t{t} = KFCS estimate of nonzero set at t
% x_upd(:,t) = x_hat at t
% Q1 = system noise covariance computed using current T_hat
% max_add: maximum number of additions allowed: ensure A(:,T_hat) remains full rank
% thresh: addition threshold
% thresh_del: deletion threshold
% k



sigma0 = sqrt(R(1,1));
sigma0_2 = R(1,1);
sig2sys = Q(1,1);
sig2init = Pi0(1,1)*3
thresh_init = (3/4)*lambdap * sigma0; %3*
thresh = 3*(1/4)*lambdap * sigma0 ;  % 3*
thresh_del = thresh/3;
thresh_Pupd = 0.05*sigma0^2;
k = 3; %duration to average x_upd to check for deletion
%To Do: try smoothing estimates.

%x_pred = zeros(m,tot);
x_upd = zeros(m,tot);
P_pred = zeros(m,m);
P_upd = zeros(m,m);
Q1 = zeros(m,m);
t=1;

max_add = floor( (1.25)*n/log2(m)); %Smax/2 %1.25*Smax;
%T_hat = T1; S_hat = length(T_hat);

%Initial CS: Running No-Noise-CS here: better when doing Regular CS
[x_hat_ls, x_hat,T_hat_tmp] = cs_solve(y(:,t),A(:, :, t),0,sigma0,lambdap,thresh_init); %change alpha
T_hat = find(abs(x_hat) > thresh_init);
    % prevent too many extra additions
if length(T_hat) > max_add 
    disp('more than max_add directions detected'),
    [val,indx] = sort(abs(x_hat)); %(T_hat)
    T_hat=indx(end:-1:end-max_add+1);
end

%Initial KF
P_pred(T_hat,T_hat) = sig2init*eye(length(T_hat));
K = P_pred*A(:, :, t)'*inv(R + A(:, :, t)*P_pred*A(:, :, t)');
P_upd  = ( eye(m) - K*A(:, :, t) )*P_pred;
xpred = zeros(m,1);
xupd = xpred + K*(y(:,t) - A(:, :, t)*xpred);
x_upd(:,t) = xupd;
T_hat_t{t} = T_hat;

for t = 2 : tot %t2   %
    % Temporary KF 
    T_hat_old = T_hat_t{t-1};
    Q1 = zeros(m,m); 
    Q1(T_hat_old,T_hat_old) = sig2sys*eye(length(T_hat_old));
    P_pred = P_upd + Q1;
    K = P_pred*A(:, :, t)'*inv(A(:, :, t)*P_pred*A(:, :, t)' + R);
    P_upd  = ( eye(m) - K*A(:, :, t) )*P_pred;
    xupd_tmp = x_upd(:,t-1) + K*(y(:,t) - A(:, :, t)*x_upd(:,t-1));

    % Compute filtering error (FE), FEN
    y_fe =  y(:,t) - A(:, :, t)*xupd_tmp;

    %CS on FE
        [betahat_ls_tmp,beta_hat,T_hat_tmp] = cs_solve(y_fe,A(:, :, t),1,sigma0,lambdap,thresh); %change alpha
        T_hat_c = setdiff([1:m]',T_hat);
        Tdiff_hat = intersect(T_hat_c, find(abs(beta_hat) > thresh) );%Sdiff_hat = length(Tdiff_hat);
        
        %preventing too many additions: to ensure A(:,T_hat) remains full rank
    %     if length(Tdiff_hat) > max_add, %1.25*
    %         Tdiff_hat0 = Tdiff_hat;
    %         disp('more than max_add directions detected'),
    %         [val,indx] = sort(abs(beta_hat));
    %         Tdiff_hat=indx(end:-1:end-max_add+1);
    %     end
        T_hat_1 = [T_hat; Tdiff_hat];
        A_T_hat_1 = A(:, T_hat_1, t);  
        len = length(Tdiff_hat);
        if (len > max_add) | (rank(A_T_hat_1) < length(T_hat_1))
            totadd = min(len,max_add);
            Tdiff_hat0 = Tdiff_hat;
            [val,indx] = sort(abs(beta_hat));
            Tdiff_hat=indx(end:-1:end-totadd+1); 
            T_hat_1 = [T_hat; Tdiff_hat];
            A_T_hat_1 = A(:, T_hat_1, t);
            while rank(A_T_hat_1) < length(T_hat_1)                
                disp('A_T_hat_1 is not full rank')
                Tdiff_hat = Tdiff_hat(1:end-1);
                T_hat_1 = [T_hat; Tdiff_hat];
                A_T_hat_1 = A(:, T_hat_1, t);
            end
        end
        T_hat = [T_hat; Tdiff_hat]; 
            
        
    % Change Q1 after CS
    Q1 = zeros(m,m); 
    diffset = setdiff(T_hat,T_hat_old);
    commonset = intersect(T_hat,T_hat_old);
    Q1(diffset,diffset) = sig2init*eye(length(diffset));
    Q1(commonset,commonset) = sig2sys*eye(length(commonset));
    
    % Final LS
    if isempty(Tdiff_hat)
    %        'empty'
        x_upd(:,t) = xupd_tmp;
    else
        A_T = A(:, T_hat_1, t);
        xupd = zeros(m,1);
        xupd(T_hat,1) = A_T \ y(:,t);
        x_upd(:,t) = xupd;
        P_upd = zeros(m,m); P_upd(T_hat,T_hat) = inv(A_T'*A_T)*sigma0_2;
    end
    T_hat_t{t} = T_hat;
%     A_T = A(:,T_hat);
%     xupd = zeros(m,1);
%     xupd(T_hat,1) = A_T \ y(:,t);
%     x_upd(:,t) = xupd;
%     P_upd = zeros(m,m); P_upd(T_hat,T_hat) = inv(A_T'*A_T)*sigma0_2;
%     T_hat_t{t} = T_hat;
    
    %COEFFICIENT REMOVAL: do it only if KF stabilized,
        %BETTER WAY: before removing check to make sure x_upd still "constant". If it is not => make a mistake earlier: put it back into the NZ set. 
    if (t>5)&(isequal(T_hat_t{t},T_hat_t{t-1},T_hat_t{t-2})) %,T_hat_t{t-3},T_hat_t{t-4}))
        tmp = find( (abs(x_upd(:,t))  < thresh_del)&(abs(x_upd(:,t-1))  < thresh_del) &(abs(x_upd(:,t-2))  < thresh_del) );
        %tmp= find( (mean(abs(x_upd(:,t-k+1:t)),2) < thresh_del) );
        Delta_c = intersect(T_hat, tmp );        
        T_hat = setdiff(T_hat, Delta_c);        
        T_hat_t{t} = T_hat;                
        Delta_r = Delta_c; %Delta_r = setdiff(T_hat_t{t-5},T_hat_t{t}) ;%intersect(T_hat_c,tmp2) % intersect(const_set, tmp2) 
        P_upd(Delta_r,[1:m]) = 0; 
        P_upd([1:m],Delta_r) = 0;
        x_upd(Delta_r,t) = 0;
        %T_hat_c = setdiff([1:m]',T_hat); tmp2 = find(diag(P_upd) < thresh_Pupd);
    end
end

%disp('change: Delta_r = Delta_c')

%if (t>4)&(isequal(T_hat_t{t},T_hat_t{t-1}))&(isequal(T_hat_t{t-1},T_hat_t{t-2}))&(isequal(T_hat_t{t-2},T_hat_t{t-3}))&(isequal(T_hat_t{t-3},T_hat_t{t-4}))

% ICIP CS-FE step
%       A_rest = A(:,T_rest);T_rest = setdiff([1:m]',T_hat);
%       [x_rest_hat_ls, x_rest_hat,T_rest_hat] = cs_solve(y_rest,A_rest,1,sigma0,lambdap,thresh); %change alpha%
%         if length(T_rest_hat) > S %S_hat
%             disp('more than S new directions detected') %keyboard
%             [val,indx] = sort(abs(x_rest_hat_ls));
%             T_rest_hat=indx(end:-1:end-S+1); %indx(end-S_hat+1:end); %
%         end
%         Tdiff_hat = T_rest(T_rest_hat);    Sdiff_hat = length(Tdiff_hat);


