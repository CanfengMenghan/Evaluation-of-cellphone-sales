B=0:0.01:23;
B=B';
B_=zeros(2301,1);
B=[B B_];
for i=1:1324
    B(fix(100*(A(i,1)))+1,2)=B(fix(100*(A(i,1))+1),2)+1;
end
count=0;
for i = 1:2301
    if B(i,2)~=0
        count=count+1;
        C(count,1)=B(i,1);
        C(count,2)=B(i,2);
    end
end