function [x, history] = ATNISD(A, b, lambda, rho, Penal)
%------------------------------------------------------------------
% The ATNISD algorithm for the following paper:
% "Robust Structure Identification of Industrial 
% Cyber-Physical System from Sparse Data: a Network Science Perspective" . 
% The algorithm adopts from the ADMM.
% Coded by: Yichi Zhang, Central South University
%------------------------------------------------------------------

% 
% Solves the following problem via ADMM:
%
%   minimize 1/2*|| Ax - b ||_2^2 + \lambda || x ||_1 + lambda2 || x ||_TV
%	subject to: x = Knn * x
%
% The solution is returned in the matrix x.
%
% history is a structure that contains the objective value, the primal and 
% dual residual norms, and the tolerances for the primal and dual residual 
% norms at each iteration.
% 
% rho is the augmented Lagrangian parameter. 
%
%

t_start = tic;

%% Global constants and defaults

QUIET    = 0;
MAX_ITER = 2000;
ABSTOL   = 1e-10;
RELTOL   = 1e-10;

%% Data preprocessing

[m, n, T] = size(A);

% save a matrix-vector multiply
for i = 1:T
    Atb(:,i) = A(:,:,i)' * b(:,i);
end


%% ADMM solver

x = zeros(n,T);
z1 = zeros(n,T);	% sparse regularization
u1 = zeros(n,T);	
z2 =zeros(n,T);		% time scale regularization
u2 =zeros(n,T);
z3 = zeros(n,T);	% space scale regularization
u3 = zeros(n,T);


% TV matrix
F = zeros(n-1,n);
for i = 1:n-1
    F(i,i) = 1;
    F(i,i+1) = -1;
end

% communication matrix calculation
Knn = Commu_maxt(sqrt(n),sqrt(n));

if ~QUIET
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
      'r norm', 'eps pri', 's norm', 'eps dual', 'r1 norm', 'eps pri1', 's1 norm', 'eps dual1','objective');
end

for k = 1:MAX_ITER
    Pm = 3 * rho * (speye(n)) +  2 * Penal * (speye(n) - Knn)'*(speye(n) - Knn);
    invPm = inv(Pm);

    %   x-update
    for i = 1:T    
        q = Atb(:,i) + rho*( (z1(:,i)-u1(:,i)) + (z2(:,i) - u2(:,i)) + (z3(:,i) - u3(:,i)));    % 
        dominator = Pm + A(:,:,i)'*A(:,:,i);
        x(:,i) = inv(dominator) * q;
    end

    zold1 = z1;
	zold2 = z2;
	zold3 = z3;
    %   z-update
    for i = 1:T
        z1(:,i) = shrinkage(x(:,i) + u1(:,i), lambda(1)/rho);
    end
    
    for i = 1:n
        xtemp = x';
        utemp = u2';
        z2temp(:,i) = TV_Condat_v2(xtemp(:,i)+utemp(:,i), lambda(2));
    end
	z2 = z2temp';
	
	for i = 1:T
		z3(:,i) = TV_Condat_v2(x(:,i) + u3(:,i), lambda(3));
	end

    %   u-update
    u1 = u1 + x - z1;
	u2 = u2 + x - z2;
	u3 = u3 + x - z3;

    
    % diagnostics, reporting, termination checks
    history.objval(k)  = objective(A, b, lambda, x, z1, z2, z3, F, Penal, Knn);

    history.r_norm(k)  = norm(x - z1);
    history.s_norm(k)  = norm(-rho*(z1 - zold1));
    history.r1_norm(k) = norm(x - z2);
    history.s1_norm(k) = norm(-rho*(z2 - zold2));
    
    history.eps_pri(k) = sqrt(n)*ABSTOL + RELTOL*max(norm(x), norm(-z1));
    history.eps_dual(k)= sqrt(n)*ABSTOL + RELTOL*norm(rho*u1);
    history.eps_pri1(k)= sqrt(n)*ABSTOL + RELTOL*max(norm(x), norm(-z2));
    history.eps_dual1(k)=sqrt(n)*ABSTOL + RELTOL*norm(rho*u2);
    
%     
%     if ~QUIET
%         fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
%             history.r_norm(k), history.eps_pri(k), ...
%             history.s_norm(k), history.eps_dual(k), ...
%             history.r1_norm(k),history.eps_pri1(k), ...
%             history.s1_norm(k),history.eps_dual1(k), ...
%             history.objval(k));
%     end
    
    if ~QUIET
        fprintf('%3d\t%10.2f\n', k,history.objval(k));
    end

    if (history.r_norm(k) < history.eps_pri(k) && ...
       history.s_norm(k) < history.eps_dual(k)&& ...
       history.r1_norm(k) < history.eps_pri1(k)&& ...
       history.s1_norm(k) < history.eps_dual1(k))
         break;
    end

end

if ~QUIET
    toc(t_start);
end

end

function p = objective(A, b, lambda, x, z1, z2, z3, F, Penal, knn)
    [m,n,T] = size(A);
	p = 0;
	z2temp = z2;
    for i = 1:T
		p = p + 1/2*norm(A(:,:,i) * x(:,i) - b(:,i),2)...
		+ lambda(1) * norm(z1(:,i))...
		+ lambda(2) * norm(F * z2temp(:,i))...
		+ Penal * norm(x(:,i) - knn*x(:,i),2)...
		+ lambda(3) *norm(F * z3(:,i));
    end
end

function z = shrinkage(x, kappa)
    z = max( 0, x - kappa ) - max( 0, -x - kappa );
end

function Matrix_K = Commu_maxt(M,N)
    SIZE = M*N;
    matrix1 = zeros(SIZE);
    matrix1(1,1)=1;
    for i=2:SIZE
        j=1+M*(i-1);
        if(j<=M*N)
            matrix1(i,j)=1;        
        else
                j = fix(j/SIZE)+mod(j,SIZE);
                matrix1(i,j)=1;             
                  
        end            
    end
    Matrix_K =matrix1;
end


% Total variation denoising of 1-D signals, a.k.a. Fused lasso
% signal approximator, by Laurent Condat.
%
% Version 2.0, Aug. 3, 2017.
%
% Given a real vector y of length N and a real lambda>=0, the 
% goal is to compute the real vector x minimizing 
%    ||x-y||_2^2/2 + lambda.TV(x), 
% where ||x-y||_2^2 = sum_{n=1}^{N} (x[n]-y[n])^2 and
% TV(x) = sum_{n=1}^{N-1} |x[n+1]-x[n]|.
function x = TV_Condat_v2(y, lambda)
	N = length(y);
	if N<=1, x=y; return; end;
	x = zeros(size(y)); % y can be a row or column vector.
	indstart_low=zeros(1,N); % starting indices of constant
		% segments of the lower approximation x_low
	indstart_up=zeros(1,N); % starting indices of constant
		% segments of the upper approximation x_up
	j_low = 1; % index to count the segments of x_low
	j_up = 1;  % same for x_up
	jseg = 1;  % segment number of the current part under
		% construction
	indjseg = 1; % starting index of the current part
	% we have indjseg = indstart_low(jseg) = indstart_up(jseg)
	indstart_low(1) = 1; % starting index of the j_low-th 
		% segment of x_low
	indstart_up(1) = 1; % same for x_up
	x_low_first = y(1)-lambda; % value of the first segment
		% of the part of x_low under construction
	x_up_first = y(1)+lambda; % same for x_up
	x_low_curr = x_low_first; % value of the last segment
		% of the part of x_low under construction
	x_up_curr = x_up_first; % same for x_up
	% the constant value of x_low over the j-th segment is stored
	% in x(indstart_low(j_low)), except for j=jseg, where the
	% value is x_low_first. Same for x_up. Indeed, the parts of 
	% x_low and x_up under construction have distinct jump 
	% locations, but same starting index jseg.
	for i = 2:N-1
	    if y(i)>=x_low_curr
	    	if y(i)<=x_up_curr
	        	% fusion of x_up to keep it nondecreasing
		        x_up_curr=x_up_curr+(y(i)-x_up_curr)/(i-indstart_up(j_up)+1);      
		        x(indjseg)=x_up_first;
		        while (j_up>jseg)&&(x_up_curr<=x(indstart_up(j_up-1)))
		        	j_up=j_up-1;
		        	x_up_curr=x(indstart_up(j_up))+(x_up_curr-x(indstart_up(j_up)))*...
		        		((i-indstart_up(j_up+1)+1)/(i-indstart_up(j_up)+1));
		        end
		        if j_up==jseg,  % a jump in x downwards is possible       	
					% the fusion of x_low has not been done yet, but this is OK.
			        while (x_up_curr<=x_low_first)&&(jseg<j_low)
			        	% the second test should always be true if the first one
			        	% is true and lambda>0, but this is a numerical safeguard.
			        	% And it is necessary if lambda=0.
			        	% validation of segments of x_low in x
				    	jseg=jseg+1;
				    	x(indjseg:indstart_low(jseg)-1)=x_low_first;
				    	x_up_curr=x_up_curr+(x_up_curr-x_low_first)*...
				    		((indstart_low(jseg)-indjseg)/(i-indstart_low(jseg)+1));
				    	indjseg=indstart_low(jseg);
				    	x_low_first=x(indjseg);
	       			end
	    			x_up_first=x_up_curr;
	    			j_up=jseg;
	    			indstart_up(jseg)=indjseg;
	        	else, x(indstart_up(j_up))=x_up_curr; end
	        else % we start a new segment in x_up
	        	j_up=j_up+1;
		        indstart_up(j_up)=i;
		        x(i)=y(i);
		        x_up_curr=x(i);
	        end
	        % fusion of x_low to keep it nonincreasing
	        x_low_curr=x_low_curr+(y(i)-x_low_curr)/(i-indstart_low(j_low)+1);      
	        x(indjseg)=x_low_first;
	        while (j_low>jseg)&&(x_low_curr>=x(indstart_low(j_low-1)))
	        	j_low=j_low-1;
	        	x_low_curr=x(indstart_low(j_low))+(x_low_curr-x(indstart_low(j_low)))*...
	        		((i-indstart_low(j_low+1)+1)/(i-indstart_low(j_low)+1));
	        end
	        if j_low==jseg  % a jump in x upwards is possible  
	        	while (x_low_curr>=x_up_first)&&(jseg<j_up)
	        		% validation of segments of x_up in x
			    	jseg=jseg+1;
			    	x(indjseg:indstart_up(jseg)-1)=x_up_first;
			    	x_low_curr=x_low_curr+(x_low_curr-x_up_first)*...
			    		((indstart_up(jseg)-indjseg)/(i-indstart_up(jseg)+1));
			    	indjseg=indstart_up(jseg);
			    	x_up_first=x(indjseg);
	       		end
	       		x_low_first=x_low_curr;
	       		j_low=jseg;
	       		indstart_low(jseg)=indjseg;
	       		if indjseg==i, % this part is not mandatory, it is a kind
	       			% of reset to increase numerical robustness.
	       			% If we are here, this is just after a jump upwards has
	       			% been validated. We have x_up_first=y(i).
	       			x_low_first=x_up_first-2*lambda;
	       		end; 
	        else, x(indstart_low(j_low))=x_low_curr; end
	    else
	    	% we start a new segment in x_low
	       	j_low = j_low+1;
	        indstart_low(j_low) = i;
	        x(i)=y(i);
	        x_low_curr=x(i);
	        % fusion of x_up to keep it nondecreasing
	        x_up_curr=x_up_curr+(y(i)-x_up_curr)/(i-indstart_up(j_up)+1);      
	        x(indjseg)=x_up_first;
	        while (j_up>jseg)&&(x_up_curr<=x(indstart_up(j_up-1)))
	        	j_up=j_up-1;
	        	x_up_curr=x(indstart_up(j_up))+(x_up_curr-x(indstart_up(j_up)))*...
	        		((i-indstart_up(j_up+1)+1)/(i-indstart_up(j_up)+1));
	        end
	        if j_up==jseg  % a jump in x downwards is possible 
	        	while (x_up_curr<=x_low_first)&&(jseg<j_low) 
	        		% validation of segments of x_low in x
			    	jseg=jseg+1;
			    	x(indjseg:indstart_low(jseg)-1)=x_low_first;
			    	x_up_curr=x_up_curr+(x_up_curr-x_low_first)*...
			    		((indstart_low(jseg)-indjseg)/(i-indstart_low(jseg)+1));
			    	indjseg=indstart_low(jseg);
			    	x_low_first=x(indjseg);
       			end
    			x_up_first=x_up_curr;
    			j_up=jseg;
    			indstart_up(jseg)=indjseg;
    			if indjseg==i, % this part is not mandatory, it is a kind
	       			% of reset to increase numerical robustness.
    				x_up_first=x_low_first+2*lambda;
    			end;
	        else, x(indstart_up(j_up))=x_up_curr; end
	    end
	end
	i=N;
	if y(i)+lambda<=x_low_curr 
		% the segments of x_low are validated
        while jseg<j_low
	    	jseg=jseg+1;
	    	x(indjseg:indstart_low(jseg)-1) = x_low_first;
	    	indjseg=indstart_low(jseg);
	    	x_low_first=x(indjseg);
     	end
     	x(indjseg:i-1) = x_low_first;
     	x(i)=y(i)+lambda;
    elseif y(i)-lambda>=x_up_curr 
		% the segments of x_up are validated
		while jseg<j_up
	    	jseg=jseg+1;
	    	x(indjseg:indstart_up(jseg)-1) = x_up_first;
	    	indjseg=indstart_up(jseg);
	    	x_up_first=x(indjseg);
     	end
     	x(indjseg:i-1) = x_up_first;
     	x(i)=y(i)-lambda;
    else
    	% fusion of x_low to keep it nonincreasing
        x_low_curr=x_low_curr+(y(i)+lambda-x_low_curr)/(i-indstart_low(j_low)+1);      
        x(indjseg)=x_low_first;
        while (j_low>jseg)&&(x_low_curr>=x(indstart_low(j_low-1)))
        	j_low=j_low-1;
        	x_low_curr=x(indstart_low(j_low))+(x_low_curr-x(indstart_low(j_low)))*...
        		((i-indstart_low(j_low+1)+1)/(i-indstart_low(j_low)+1));
        end
        if j_low==jseg % the segments of x_up must be validated
        	if x_up_first>=x_low_curr % same unique segment of x_low and x_up
        		x(indjseg:i)=x_low_curr;
        	else
				% fusion of x_up to keep it nondereasing
        		x_up_curr=x_up_curr+(y(i)-lambda-x_up_curr)/(i-indstart_up(j_up)+1);      
	        	x(indjseg)=x_up_first;
	        	while (j_up>jseg)&&(x_up_curr<=x(indstart_up(j_up-1)))
	        		j_up=j_up-1;
	        		x_up_curr=x(indstart_up(j_up))+(x_up_curr-x(indstart_up(j_up)))*...
	        			((i-indstart_up(j_up+1)+1)/(i-indstart_up(j_up)+1));
	        	end
	       		x(indstart_up(j_up):i)=x_up_curr;
        		while jseg<j_up  % the segments of x_up are validated
		    		jseg=jseg+1;
		    		x(indjseg:indstart_up(jseg)-1) = x_up_first;
		    		indjseg=indstart_up(jseg);
		    		x_up_first=x(indjseg);
	     		end
	     	end
        else 	% the segments of x_low must be validated
        	x(indstart_low(j_low):i)=x_low_curr;
        	while jseg<j_low
		    	jseg=jseg+1;
		    	x(indjseg:indstart_low(jseg)-1) = x_low_first;
		    	indjseg=indstart_low(jseg);
		    	x_low_first=x(indjseg);
	     	end
        end
    end
end	