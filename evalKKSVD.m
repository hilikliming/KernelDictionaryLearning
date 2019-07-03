function [r] = evalKKSVD(z,B,A,T0,kfnc,KBB,X)

option.kernel = 'cust'; option.kernelfnc=kfnc;
Kzz=computeKernelMatrix(z,z,option);
KzY=computeKernelMatrix(z,B,option);
%KYY=computeKernelMatrix(Y,Y,option);
for q=1:size(z,2)
    if nargin<7
        [x] = KOMP(z(:,q),B,A,T0,kfnc,KBB);
    else
        x=X(:,q);
    end
    
    r(q)=Kzz(q,q)-2*KzY(q,:)*A*x+x'*A'*KBB*A*x;
end
end

