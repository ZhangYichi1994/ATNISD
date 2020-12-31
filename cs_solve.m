% sigma = 0 ; %run CS for no noise case: assume small noise is actually the "compressible" part of a compr. signal x.
function [xhat_ls,xhat,T_hat] = cs_solve(y,A,sigma,sigma0,lambdap,alpha)
[n,m] = size(A);
%if sigma=0, need not provide sigma0 and lambdap
if sigma>0
    del = lambdap * sigma0 ;
    c = [zeros(m,1);
        ones(m,1)];
    F = [eye(m,m)  , -eye(m,m) ;
        -eye(m,m) , -eye(m,m) ;
        A'*A      , zeros(m,m);
        -A'*A     , zeros(m,m);];       %-A'A*x < =
    %zeros(m,m), -eye(m,m) ]  ;    %u >= 0
    b = [zeros(m,1);
        zeros(m,1);
        del + A'*y;
        del - A'*y;];
    %zeros(m,1)];
    [xfullhat,fval,exitflag,output,lambda] = linprog(c,F,b);
else
    c = [zeros(m,1);
        ones(m,1)];
    F_exact = [ eye(m,m)  , -eye(m,m) ;
        -eye(m,m) , -eye(m,m) ;
        zeros(m,m), -eye(m,m) ];
    b_exact = [zeros(m,1);
        zeros(m,1);
        zeros(m,1)];
    F_eq_exact = [A, zeros(n,m)];
    b_eq_exact = y;
    [xfullhat,fval,exitflag,output,lambda] = linprog(c,F_exact,b_exact,F_eq_exact,b_eq_exact);
end
xhat = xfullhat(1:m);     uhat = xfullhat(m+1:end);
if exist('alpha')~=1
    if sigma > 0
        alpha = (0.5/4)*lambdap * sigma0; %0.5
    else
        alpha = (3/4)*lambdap * sigma0;
    end
end
T_hat = find(abs(xhat) > alpha);
AT_hat = A(:,T_hat);
xhat_ls_T_hat = inv(AT_hat'*AT_hat) * AT_hat'*y;
xhat_ls = zeros(size(xhat));
xhat_ls(T_hat) = xhat_ls_T_hat;
