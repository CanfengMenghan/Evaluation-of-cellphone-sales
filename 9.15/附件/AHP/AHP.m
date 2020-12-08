%clear B
jitiaoshuju=size(A);
jitiaoshuju=jitiaoshuju(1,1);
R=A(:,1);
xingbie=max(R);
R=A(:,2);
jigecanhe=max(R);
%A是原始数据
for b=1:xingbie%性别
B{b}=[];
end
for c=1:jitiaoshuju%几条数据
for d=1:xingbie%性别
if A(c,1)==d
B{d}=[B{d};A(c,:)];
end
end
end
for e=1:xingbie%性别
for f=1:jigecanhe%几个餐盒
T=B{e}(:,2);
Q=find(T(T==f));
U(e,f)=max(Q);
end
end
%U是原始个数，比如第二行第一列就是女的餐盒选项为1的有多少个人
for l=1:xingbie%有几个性别
for j=1:jigecanhe%有几个餐盒
for k=1:jigecanhe%有几个餐盒
C{l}(j,k)=U(l,j)/U(l,k);
end
end
end
for i=1:xingbie%有几个性别
t=C{i};
[x,lumda]=eig(t); 
r=abs(sum(lumda)); 
n=find(r==max(r)); 
max_lumda_A(1,i)=lumda(n,n); 
max_x_A{i}=x(:,n); %特征值
max_x_A{i}=max_x_A{i}./sum(max_x_A{i});
end
for p=1:xingbie%有几个性别
for q=1:jigecanhe%有几个餐盒
max_x_AB(p,q)=max_x_A{p}(q,1);%特征向量，第一行是男的的各项权重，第二行是女的的各项权重
end
end