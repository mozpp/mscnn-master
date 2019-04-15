function a()
image_dir = '../dataset/images/';
image_list = dir([image_dir '*.jpg']);
file=fopen('train_labels.csv');
resize_w = 1408; resize_h = 1088; Min_Height = 1;
car1_width=[];car1_hight=[];
ratio_w = resize_w/2666.0;
ratio_h = resize_h/2000.0;
while ~feof(file)
    tline=fgetl(file);
    tline=textscan(tline,'%s ','delimiter',', ');
    
    w_i=(str2double(tline{1,1}{4,1}))-(str2double(tline{1,1}{2,1}));
    h_i=(str2double(tline{1,1}{5,1}))-(str2double(tline{1,1}{3,1}));
    w=w_i*ratio_w; h=h_i*ratio_h;
    if  h>Min_Height,
        car1_width=[car1_width;w];
        car1_hight=[car1_hight;h];
    end
end
car_num1 = length(car1_width);

sort_width1 = sort(car1_width);
sort_hight1 = sort(car1_hight);
% sort_width2 = sort(car2_width);
% sort_height2 = sort(car2_hight);
% ini_width1=[];ini_hight1=[];ini_width2=[];ini_hight2=[];
K=7;
% ini_width1=get_centroid(car1_width,K);

ini_width1=[];ini_hight1=[];
for i=1:K
    ini_width1=[ini_width1;sort_width1(ceil(car_num1/(K+1)*i))];
    
    ini_hight1=[ini_hight1;sort_hight1(ceil(car_num1/(K+1)*i))];
end

[Idx,field1_w,sumD,D]=kmeans(car1_width,K,'Start',ini_width1);%[60;84;120;168;240;336;480]
for i=1:K
    fprintf('field1_w %d: %d\n',i,round(field1_w(i)));
end
[Idx,field1_h,sumD,D]=kmeans(car1_hight,K,'Start',ini_hight1);
for i=1:K
    fprintf('field1_h %d: %d\n',i,round(field1_h(i)));
end

for i=1:K
    fprintf('field_w %d: %f\n',i,field_w(i));
end
for i=1:K
    fprintf('field_h %d: %f\n',i,field_h(i));
end

end%main
% 
function cluster_centers=get_centroid(points,K)
%%get centroids
m=length(points);
cluster_centers=zeros(K,1);
cluster_centers(1)=randi(m);
for i=1:K
    sum_all=0;
    
   for j=1:m
       d(j) = nearest(points(j), cluster_centers(1:i));
       sum_all = sum_all+d(j);
   end
   sum_all=sum_all*rand;%random 0-1
   for j=1:m
       sum_all=sum_all-d(j);
       if sum_all>0,
           continue
       end
       cluster_centers(i)=points(j);
       break
   end
end

end
function d=nearest(point,cluster_centers)
min_dist=1e100;
d=min(abs(point-cluster_centers));

end