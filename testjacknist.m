% Opening custom characters
clc
clear all
indices=(15:29:278)+1;%[20,48,78,109,134,160,184,216,244,272];
labelsjack=[0,1,2,3,4,5,6,7,8,9];
X=imread('jacknist.png');
X=double(X(:,:,1));
testX=[];
ii=1;

%X=-X+255;
for idx = indices
    figure;
    X(end,idx-14:idx+13)=0;
    imgZ=X(:,idx-13:idx+14);
    
    imgZ=imresize(imgZ,[20,20]);
    imgZ=[zeros(4,size(imgZ,2));imgZ;zeros(4,size(imgZ,2))];
    imgZ=[zeros(size(imgZ,1),4),imgZ,zeros(size(imgZ,1),4)];
    Ilabel = bwlabel(imgZ);
    stat = regionprops(Ilabel,'centroid');
    imgZ=circshift(imgZ,floor(14-stat.Centroid(2)),1)
    imgZ=circshift(imgZ,floor(14-stat.Centroid(1)),2);
    
    testX(:,ii)=reshape(imgZ',[],1);
    imagesc(reshape(testX(:,ii),28,28));
    colormap('bone');
    ii=ii+1;
end
testX


testX=normc(testX);
testZ=testX;
testY=labelsjack;
M=length(unique(testY));
Resid=zeros(M,size(testZ,2));
sigma=3;
kfncs{1} = @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/(2*sigma))); %@(x,y) (x'*y);%  Gaussian Kernel
load('mnist_model_nokmeans.mat');
T0=5;

for m =1:M
    Resid(m,:)=evalKKSVD(testZ,Bs(m).B,As(m).A,T0,kfncs{1},Kyy(m).KYY);
end

for s=1:size(Resid,2)
    [~,testYhat(s)]=min(Resid(:,s));
end

%%

pltcnf(testY+1,testYhat,1)


