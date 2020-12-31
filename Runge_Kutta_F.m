function [x1,y1,z1]=Runge_Kutta_F(x,y,z,X,Row,AA,C,h,N)
%          (x,y,z) -- 输入初始点
%                X -- 阵子矩阵
%              ROW -- 网络节点选择
%               AA -- 内联矩阵
%                C -- 邻接矩阵
[k1,l1,p1]=Lorenz_Net(x,y,z,X,Row,AA,C,N);
[k2,l2,p2]=Lorenz_Net(x+h/2*k1,y+h/2*l1,z+h/2*p1,X,Row,AA,C,N);
[k3,l3,p3]=Lorenz_Net(x+h/2*k1,y+h/2*l1,z+h/2*p1,X,Row,AA,C,N);
[k4,l4,p4]=Lorenz_Net(x+h*k1,y+h*l1,z+h*p1,X,Row,AA,C,N);
x1=x+h/6*(k1+2*k2+2*k3+k4);
y1=y+h/6*(l1+2*l2+2*l3+l4);
z1=z+h/6*(p1+2*p2+2*p3+p4);
end

%洛伦兹网络系统
function [u,v,w]=Lorenz_Net(x,y,z,X,Row,AA,C,N)
num_nodes = N;
u=10*(y-x);
v=28*x-y-x*z;
w=x*y-8/3*z;
Q=[u,v,w]';
Q1 = Q;
  for g=1:num_nodes
     E=Q+C(Row,g)*AA*X(:,g);
     Q=E;
  end
[u,v,w]=deal(E(1),E(2),E(3));   

end

