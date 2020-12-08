%yangben第一列是分类敲进去
%b是待判的敲进去，g敲进去
%iii是概率，结果
%H是后验概率，结果
%g-group分类数，后来写了个自动检测分类数的，不过没在matlab下些，呵呵
[m,n]=size(yangben);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:g
groupNum(i)=0;
group(i)=0;
for j=1:m
if yangben(j,1)==i
group(i)=group(i)+1;
end
end
if i==1
groupNum(i)=group(i);
else
groupNum(i)=groupNum(i-1)+group(i);
end
end
group;
groupNum; %计算分类个数数组
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%计算总平均值
% for j=1:n-1
% TotalMean(j)=0;
% for i=1:m
% TotalMean(j)=TotalMean(j)+yangben(i,j+1);
% end
% TotalMean(j)=TotalMean(j)/m;
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
GroupMean=[];
for i=1:g
if i==1
low=1;
up=groupNum(i);
else
low=groupNum(i-1)+1;
up=groupNum(i);
end
matrix=yangben(low:up,:);
MatrixMean=mean(matrix); %各分类组平均值
GroupMean=[GroupMean;MatrixMean];

for u=low:up
for v=2:n
C(u,v-1)=yangben(u,v)-MatrixMean(v);
end
end
end

C;
GroupMean;
V=C'*C/(m-g);
V_inv=inv(V); %对矩阵V求逆
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
GroupMean=GroupMean(:,2:n);
Q1=GroupMean*V_inv;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:g
lnqi(i)=log(group(i)/m);
mat=GroupMean(i,:);
Q2(i)=lnqi(i)-0.5*mat*V_inv*mat';
end
lnqi;
Q2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[u,v]=size(b);
result=[];
for i=1:u
x=b(i,:);
yy=Q1*x'+Q2';
result=[result yy];
end
res=result'; %计算的待判数据对各标准数据的线性计算值

[rows,cols]=size(result);
for i=1:cols
iljj=0;
mlljj=result(:,i);
for j=1:rows
iljj=iljj+exp(result(j,i)-max(mlljj));
end
for j=1:rows
houyangailv(j,i)=exp(result(j,i)-max(mlljj))/iljj;
end
end
H=houyangailv'; %后验概率

iii=[];
for a=1:u
k=max(H(a,:));
for ii=1:g
if k==H(a,ii)
iii=[iii;ii];
end
end
end