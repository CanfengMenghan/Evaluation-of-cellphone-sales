B=0:1:15;
B=B';
B_=zeros(16,1);
B=[B B_];
for i=1:2290
    B((A(i,1)+1),2)=B((A(i,1)+1),2)+1;
end
count=0;
for i = 1:16
    if B(i,2)~=0
        count=count+1;
        C(count,1)=B(i,1);
        C(count,2)=B(i,2);
    end
end