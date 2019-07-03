function [xf] = KOMP(z,B,A,T0,kfnc,KBB)



I=[];
xf=zeros(size(A,2),1);
zhat=zeros(size(B,2),1);
s=0; 

option.kernel = 'cust'; option.kernelfnc=kfnc;    
if nargin<6
    KBB=computeKernelMatrix(B,B,option);
end
% if isempty(KYY)
%     
% end

KzY=computeKernelMatrix(B,z,option)';
B=[];
C=B;
D=B;
P=[];
for t=1:T0
    
    Atemp=A;
    Atemp(:,ismember(1:size(A,2),I))=0;
    tau = (KzY-zhat'*KBB)*Atemp;
    [~,imax]=nanmax(abs(tau)); 
    I = [I,imax];
    if t==1
        P=A(:,I)'*KBB*A(:,I);
        Pinv=1./P;
    else
        B= A(:,I(1:t-1))'*KBB*A(:,I(t));
        C= A(:,I(t))'*KBB*A(:,I(1:t-1));
        D= A(:,I(t))'*KBB*A(:,I(t));
        P= [P,B;C,D];
        DCAB=(D-C*Pinv*B)^(-1);
        Pinv=[Pinv+Pinv*B*DCAB*C*Pinv,-Pinv*B*DCAB; -DCAB*C*Pinv, DCAB];   
    end
    
    xs=Pinv*(KzY*A(:,I))';
    zhat= A(:,I)*xs;

end
xf(I)=xs;
end

