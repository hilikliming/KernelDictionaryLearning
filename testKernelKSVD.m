% This is a test script for Kernel KSVD
clc
clear all
% Importing MNIST data
mnist=true;%false;%
profile on
if mnist
    d = load('mnist.mat');

    d.testX=double(d.testX)';
    d.trainX=double(d.trainX)';
    d.testX=normc(d.testX);
    d.trainX=normc(d.trainX);
    
    X = d.trainX;
    
    i = reshape(X(:,200), 28, 28)';
    imagesc(i); colormap('bone');
else
    X=randn(2,8000);
    X= X-min(min(X));
    X= X./max(max(X));
    X=X-0.5;
    for q=1:size(X,2)
       Y(1,q)= (X(1,q)>0 && X(2,q)<=0) || (X(1,q)<=0 && X(2,q)>0); 
    end
    
    d.trainY=Y(1,1:end/2);
    d.trainX=X(:,1:end/2);
    d.testY=Y(1,end/2+1:end);
    d.testX=X(:,end/2+1:end);
end

% Train Kernel K-SVD Dictionary 
M=length(unique(d.trainY));%2;%10;
Xs=struct([]);
As=struct([]);
Kyy=struct([]);
Zs=struct([]);
Ys=struct([]);
Bs=struct([]);
BYs=struct([]);

trainY=double(d.trainY);
trainZ=double(d.trainX);



use_less_samples=1;
Q=size(trainZ,2);
if use_less_samples
     ds_fac=0.15;
     
     pickInd=randperm(Q,floor(ds_fac*Q));
     trainY=trainY(pickInd);
     trainZ=trainZ(:,pickInd);
end


use_kmeans=1;
trainYk=[];
trainZk=[];
if use_kmeans & mnist
    for m=1:M
     trainZm=trainZ(:,trainY==m-1);
     Q=size(trainZm,2);
     [idx,C] = kmeans(trainZm',floor(.05*Q));
     trainYk=[trainYk,(m-1)*ones(1,size(C,1))];
     trainZk=[trainZk,C'];
    end
    %trainZ=trainZk;
    %trainY=trainYk;
else
    trainZk=trainZ;
    trainYk=trainY;
end



if mnist
    T0 =  5;
    K  =  200;
    maxIter=10;
    sigma=1;%3;%
else
    T0 =  4; %3;% 
    K  =  250; 
    maxIter = 80;%100; %
    sigma=.1;
end

for m=1:M
    Zs(m).Z = trainZ(:,trainY==m-1);
    Bs(m).B = trainZk(:,trainYk==m-1);
    BYs(m).Y= (m-1)*ones(1,size(Bs(m).B,2));
    Ys(m).Y =(m-1)*ones(1,size(Zs(m).Z,2));
end

kfncs{1} = @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/(2*sigma))); %@(x,y) (x'*y);%  Gaussian Kernel

%%
learn_models=1;

if learn_models
    parfor m = 1:M
        [As(m).A,Xs(m).X,Kyy(m).KYY] = KernelKSVD(Bs(m).B,Bs(m).B,T0,K,kfncs{1},maxIter);
    end
    profile viewer
else
    if mnist
        load('mnist_model.mat');
    else
        load('xor_model.mat');
    end
    
end
save_models=0;

if save_models
    if mnist
        save('mnist_model_nokmeans.mat','As','Kyy','Xs','Bs');
    else
        save('xor_model.mat','As','Kyy','Xs','Bs');
    end
    
end

%%
% Classifying Testing data

testZ=double(d.testX);
testY=double(d.testY);

%testZ=circshift(testZ,2,1);

Resid=zeros(M,size(testZ,2));
for m =1:M
    Resid(m,:)=evalKKSVD(testZ,Bs(m).B,As(m).A,T0,kfncs{1},Kyy(m).KYY);
end

for s=1:size(Resid,2)
    [~,testYhat(s)]=min(Resid(:,s));
end

%
if ~mnist
    figure;
    pltcnf(testY,testYhat-1,1)

    figure; hold on;
    plot(testZ(1,testY==0),testZ(2,testY==0),'b*');
    plot(testZ(1,testY==1),testZ(2,testY==1),'go');

    plot(testZ(1,testY~=testYhat-1),testZ(2,testY~=testYhat-1),'rx');
    title('Kernel K-SVD SRC on XOR Problem');
    
    num_correct=testY==testYhat-1;
    
else
    pltcnf(testY+1,testYhat,1)
    num_correct=testY==testYhat-1;
    
end
per_correct=sum(num_correct)/length(num_correct)

