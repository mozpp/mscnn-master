function anchor2()
image_dir = '../dataset/images/';
image_list = dir([image_dir '*.jpg']);
file=fopen('GT_subclass_new.txt');
resize_w = 683; resize_h = 384; Min_Height = 10;
car1_width=[];car1_hight=[];car2_width=[];car2_hight=[];car_w_h=[];
car_w_h1=[];car_w_h2=[];
while ~feof(file)
    tline=fgetl(file);
    tline=textscan(tline,'%s ') ;
    car_num=str2double(tline{1,1}{2,1});
    img_name=tline{1,1}{1,1};
    x_ori=imread([image_dir img_name '.jpg']);
    
    [hight width ~]=size(x_ori);
    ratio_w = resize_w/width;
    ratio_h = resize_h/hight;
    i=1;
    
    while i<car_num*5
        class=int32(str2double(tline{1,1}{2+i,1}));
        x_i=(str2double(tline{1,1}{3+i,1}));
        y_i=(str2double(tline{1,1}{4+i,1}));
        w_i=(str2double(tline{1,1}{5+i,1}));
        h_i=(str2double(tline{1,1}{6+i,1}));
        i=i+5;
        w=w_i*ratio_w; h=h_i*ratio_h;
        %car_w_h=[car_w_h;w,h];
        if class==1&& h>Min_Height,
            car_w_h1=[car_w_h1;w,h];
        end
        if class==2&& h>Min_Height,
            car_w_h2=[car_w_h2;w,h];
        end
    end
end

K=7;
[car1_num ~]=size(car_w_h1);[car2_num ~]=size(car_w_h2);
ini1=get_centroid(car_w_h1,K);
ini2=get_centroid(car_w_h2,K);

car_w_h1=sortrows(car_w_h1);
car_w_h2=sortrows(car_w_h2);
for i=1:K
    ini1(i,:)=car_w_h1(ceil(car1_num*i/K),:);
    ini2(i,:)=car_w_h2(ceil(car2_num*i/K),:);
end

% [Idx,field1_w,sumD,D]=kmeans(car_w_h,K,'Start',[60,60;84,84;120,120;168,168;240,240;336,336;480,480]);
% for i=1:K
%     fprintf('field1_w %d: %f\n',i,field1_w(i));
% end
[Idx,wh1,sumD,D]=kmeans(car_w_h1,K,'Start',ini1,'MaxIter',10000);
[Idx,wh2,sumD,D]=kmeans(car_w_h2,K,'Start',ini2,'MaxIter',10000);

wh=wh1*car1_num/(car1_num+car2_num)+wh2*car2_num/(car1_num+car2_num);
wh=sortrows(wh);
save wh
for i=1:K
    fprintf('w_h_%d: %.3f %.3f\n',i,wh(i,:));
end
end

function cluster_centers=get_centroid(points,K)
%%get centroids
[m,n]=size(points);
cluster_centers=zeros(K,n);
cluster_centers(1,:)=points(randi(m),:);
for i=1:K
    sum_all=0;
    
   for j=1:m
       d(j) = nearest(points(j,:), cluster_centers(1:i,:));
       sum_all = sum_all+d(j);
   end
   sum_all=sum_all*rand;%random 0-1
   for j=1:m
       sum_all=sum_all-d(j);
       if sum_all>0,
           continue
       end
       cluster_centers(i,:)=points(j,:);
       break
   end
end

end
function d=nearest(point,cluster_centers)
dbstop if error
min_dist=1e100;
d=min((point(1)-cluster_centers(:,1)).^2 + (point(2)-cluster_centers(:,2)).^2);
d=sqrt(d);
end