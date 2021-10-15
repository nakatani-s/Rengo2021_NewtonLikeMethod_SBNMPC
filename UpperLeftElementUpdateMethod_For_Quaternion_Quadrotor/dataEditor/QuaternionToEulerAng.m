clear all

dir_path = "DataToSICE_Journal/"; 
filestr = "data_state_1013_1621";
extensions1 = ".txt";
extensions2 = ".csv";
f_nameTXT = append(filestr,extensions1);
filename = fopen(f_nameTXT,"r");
% filename = fopen("Large_Unexpected_state_Proposed.txt","r");
dataFormat = '%lf';
DatafromFile = fscanf(filename, dataFormat);
row = 1000;
column = 14;
Data = zeros(row,column);

for i = 1:row
    for k = 1:column
        index = (i-1)*column+k;
        Data(i,k) = DatafromFile(index);
    end
end

% q0 = zeros(1,1200);
q0 = Data(:,8);
q1 = Data(:,9);
q2 = Data(:,10);
q3 = Data(:,11);

% for i = 1:1200
%     pow_q1 = q1(i) * q1(i);
%     pow_q2 = q2(i) * q2(i);
%     pow_q3 = q3(i) * q3(i);
%     pow_q0 = 1 - pow_q1 - pow_q2 - pow_q3;
%     q0(i) = sqrt(pow_q0);
% end

roll = zeros(1,row);
pitch = zeros(1,row);
yaw = zeros(1,row);

for i = 1:row
    q0q0 = q0(i)*q0(i);
    q0q1 = q0(i)*q1(i);
    q0q2 = q0(i)*q2(i);
    q0q3 = q0(i)*q3(i);
    q1q1 = q1(i)*q1(i);
    q1q2 = q1(i)*q2(i);
    q1q3 = q1(i)*q3(i);
    q2q2 = q2(i)*q2(i);
    q2q3 = q2(i)*q3(i);
    q3q3 = q3(i)*q3(i);
    
    roll(1,i) = atan2(2.0*(q2q3+q0q1), q0q0-q1q1-q2q2+q3q3);
    pitch(1,i) = asin(2.0*(q0q2-q1q3));
    yaw(1,i) = atan2(2.0*(q1q2+q0q3),q0q0+q1q1-q2q2-q3q3);
end

d_roll = roll*180/pi;
d_pitch = real(pitch)*180/pi;
d_yaw = yaw*180/pi;
resizedData = [Data transpose(d_roll) transpose(d_pitch) transpose(d_yaw)];

f_nameCSV = append(dir_path,filestr,extensions2);
csvwrite(f_nameCSV, resizedData);
