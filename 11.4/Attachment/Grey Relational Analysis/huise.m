[m,n]=size(A);
[p,q]=size(Z);
for j =1:n
    B{j}=A(:,j);
    B{j}=B{j}/std(B{j});

    for i=1:(m-1)
        B_{j}(i)=B{j}(i)-B{j}(i+1);
    end
end
for j =1:q
    Y{j}=Z(:,j);
    Y{j}=Y{j}/std(Y{j});

    for i=1:(p-1)
        Y_{j}(i)=Y{j}(i)-Y{j}(i+1);
    end
end
for i=1:q
    aver(i,1)=mean(Y_{i});
    for j = 1:n
        aver(i,(j+1))=abs(sum(Y_{i})-sum(B_{j}));
        final(i,j)=(1+aver(i,1))/(1+aver(i,1)+aver(i,(j+1)));
    end
end
%averv0=mean(x0_);
%averv1=averv1/1323;
%final=(1+averv0)/(1+averv0+averv);