A = [130 146 133 95 71 71 62 78; 130 146 133 92 62 71 62 71; 139 146 146 120 62 55 55 55; 139 139 139 146 117 112 117 110; 139 139 139 139 139 139 139 139; 146 142 139 139 139 143 125 139; 156 159 159 159 159 146 159 159; 168 159 156 159 159 159 139 159];
B = zeros(8);
C = [];
D = [];
for i=1:64
    if A(i) >= 128 
        B(i) = 1;
        D = [D, A(i)];
    else
        C = [C, A(i)];
    end
end

B
A(A < 125) = 0;
A(A >= 125) = 1;
A

disp(mean(C));
disp(std(C));
disp(mean(D));
disp(std(D));

x = -30:0.001:260;
y = normpdf(x,mean(C),std(C));
z = normpdf(x,mean(D),std(D));
plot (x,y,x,z);



