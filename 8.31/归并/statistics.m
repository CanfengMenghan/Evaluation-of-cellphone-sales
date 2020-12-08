clear result
result=zeros(5,5);
for i=1:1324
    if A(i,1)<75

        result(1,B(i,1))=result(1,B(i,1))+1;
    end
    if A(i,1)<150 && A(i,1)>=75
        result(2,B(i,1))=result(2,B(i,1))+1;
    end
    if A(i,1)<225 && A(i,1)>=150
        result(3,B(i,1))=result(3,B(i,1))+1;
    end
    if A(i,1)>=225 && A(i,1)<300
        result(4,B(i,1))=result(4,B(i,1))+1;
    end
    if A(i,1)>=300
        result(5,B(i,1))=result(5,B(i,1))+1;
    end
end