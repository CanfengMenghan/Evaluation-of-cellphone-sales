clear c catagory detection i j k m n 
for i=1:10
    c{i}=[];
end
for i=1:442
    catagory=b(i,1);
    for j =1:10
        [m,n]=size(c{j});
        if n~=0
            detection=0;
            for k =1:n
                
                if c{j}(5,k)==a(i,j)
                    c{j}(catagory,k)=c{j}(catagory,k)+1;
                    detection=1;
                end
            end
            if detection==0
                c{j}(5,(n+1))=a(i,j);
                c{j}(catagory,(n+1))=(c{j}(catagory,(n+1)))+1;
            end
        
        end
        if n==0
            c{j}(5,1)=a(i,j);
            c{j}(catagory,1)=(c{j}(catagory,1))+1;
        end
    end
end
for i=1:10
    c{i}=c{i}';
end

        