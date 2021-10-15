clear all

input_dim = 4;
horizonal_dim = 14;
dim = input_dim * horizonal_dim;

fname_before = fopen("HessianMatrix_1013_169_0step.txt","r");
fname_after = fopen("HessianMatrix_1013_169_749step.txt","r");
dataFormat = '%lf';

Bmat = fscanf(fname_before, dataFormat);
Gmat = fscanf(fname_after, dataFormat);


B = zeros(dim,dim);
G = zeros(dim,dim);
SHM = zeros(dim,dim);
% X = zeros(dim*dim,1);
% Y = zeros(dim*dim,1);

[X Y] = meshgrid(1:dim,1:dim);

for i = 1:dim
    for k = 1:dim
        elind = (i-1)*dim+k;
%         X(elind,1) = i;
%         Y(elind,1) = k;
        G(i,k) = Gmat(elind,:)/2.0;
        B(i,k) = Bmat(elind,:)/2.0;
        if k==i && i < dim
            SHM(i,k+1)= 1.0;
        elseif k==i && i==dim
            SHM(i,k) = 1.0;
        end
    end
end

InputDimSHM = SHM*SHM*SHM*SHM;
tr_SHM = transpose(SHM);
InputDimShiftedSHM = tr_SHM*tr_SHM*tr_SHM*tr_SHM; 
% Shift_B = SHM*B*tr_SHM;
Shift_B = InputDimSHM*B*InputDimShiftedSHM;

reshape_H = zeros(dim,dim);
for i = 1:dim
    reshape_H(dim-(i-1),:) = B(i,:); 
end
differM = G-B;
differInvM = inv(G)-inv(B);
differShiftM = G-Shift_B;

%%plot 3d-figures
figure(1);
surf(X,Y,abs(G));
caxis([0 5*10^-4])
colorbar;
view(90,90);

figure(2);
surf(X,Y,abs(differM));
caxis([0 1*10^-4])
colorbar;
view(90,90);

figure(3);
surf(X,Y,abs(differShiftM));
caxis([0 1*10^-4]);
colorbar;
view(90,90);

figure(4);
surf(X,Y,abs(differInvM));
caxis([0 10]);
colorbar;
view(90,90);