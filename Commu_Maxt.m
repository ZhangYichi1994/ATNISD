function Matrix_K = Commu_Maxt(M,N)
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