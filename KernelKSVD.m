function [Anew,X,KBB] = KernelKSVD(Y,B,T0,K,kfnc,maxIter)
% the kernel K-SVD algorithm
errThresh=1e-6;
option.kernel = 'cust'; option.kernelfnc=kfnc;
X= zeros(K,size(Y,2));
for col=1:size(Y,2)
    indon=randperm(size(X,1),T0);
    X(indon,col)=1;
end

A=normc(randn(size(B,2),K));%X'*(X*X')^(-1);
Anew=A;
KBB=computeKernelMatrix(B,B,option);

for J=1:maxIter
    disp(J);
    A=Anew;
    % SPARSE CODING STAGE
    parfor q =1:size(Y,2)
        [X(:,q)] = KOMP(Y(:,q),B,A,T0,kfnc,KBB);
    end
    
    X(isnan(X))=0;
    X(isinf(X))=0;
    
    % A UPDATE STAGE
    for k = 1:K
        drop_k=1:K;drop_k(k)=[];
        wk=find(X(k,:));
        Ek=(eye(size(Y,2))-A(:,drop_k)*X(drop_k,:));
        EkR=Ek(:,wk);
        [U,S,~]=svd(EkR'*KBB*EkR,'econ');
        if isempty(S)
           Anew(:,k)=0;
        else
           Anew(:,k)=1/sqrt(S(1,1))*EkR*U(:,1);
        end
        
    end
    
    [r] = evalKKSVD(Y,Y,Anew,T0,kfnc,KBB,X);
    if mean(abs(r))<=errThresh
        break
    end
    
    
end


end

