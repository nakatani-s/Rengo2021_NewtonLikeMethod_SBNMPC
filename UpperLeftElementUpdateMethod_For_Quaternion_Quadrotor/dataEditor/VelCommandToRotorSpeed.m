clear all

dir_path = "DataToSICE_Journal/";
filestr = "Large_Unexpected_input_MCMPC";
extensions1 = ".txt";
extensions2 = ".csv";
f_nameTXT = append(filestr,extensions1);
filename = fopen(f_nameTXT,"r");
% filename = fopen("PathFollowing_input_Proposed.txt","r");
dataFormat = '%lf';
DatafromFile = fscanf(filename, dataFormat);
row = 1000;
column = 13;
Data = zeros(row,column);

for i = 1:row
    for k = 1:column
        index = (i-1)*column+k;
        Data(i,k) = DatafromFile(index);
    end
end

% q0 = zeros(1,1200);
vz = Data(:,2);
vwx = Data(:,3);
vwy = Data(:,4);
vwz = Data(:,5);

r_cw1 = zeros(1,row);
r_cw2 = zeros(1,row);
r_cw3 = zeros(1,row);
r_cw4 = zeros(1,row);
ug = 150;
for i = 1:row
    r_cw1(i) = ug + vz(i)-vwy(i)+vwz(i);
    r_cw2(i) = ug + vz(i)+vwy(i)+vwz(i);
    r_cw3(i) = ug + vz(i)+vwx(i)-vwz(i);
    r_cw4(i) = ug + vz(i)-vwx(i)-vwz(i);
end

reData = [Data transpose(r_cw1) transpose(r_cw2) transpose(r_cw3) transpose(r_cw4)];
f_nameCSV = append(dir_path, filestr, extensions2);
csvwrite(f_nameCSV, reData);