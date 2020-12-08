[heng,shu]=size(H);
for i=1:heng
    H_(i,1)=H(i,1)/(H(i,1)+H(i,3));
    H_(i,2)=H(i,3)/(H(i,1)+H(i,3));
end