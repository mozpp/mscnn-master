% clear and close everything
clear all; close all;
disp('======= KITTI DevKit Demo =======');

root_dir  = '/media/b3-542/d/car/KITTI';
label_dir = fullfile(root_dir,'/data_object_image_2/train/label_2');
img_dir = fullfile(root_dir,'/data_object_image_2/val');
img_save = fullfile('/media/b3-542/d/JPEGImages');
test_dir  = fullfile('/media/b3-542/d/label');

% read objects of first training image
%train_objects = readLabels(label_dir,3);
mean_image = [124,117,104];

% loop over all images
% ... YOUR TRAINING CODE HERE ...
% ... YOUR TESTING CODE HERE ...

% detect one object (car) in first test image
% test_objects(1).type  = 'Car';
% test_objects(1).x1    = 10;
% test_objects(1).y1    = 10;
% test_objects(1).x2    = 100;
% test_objects(1).y2    = 100;
% test_objects(1).alpha = pi/2;
% test_objects(1).score = 0.5;

% write object to file
for i=6001:7480
    %i = 5316;
    arr = [];
    label = readLabels(label_dir,i);

    file = sprintf('%06d.png',i);
    img_path = fullfile(img_dir,file);
    img = imread(img_path);
   
    
    for j=1:length(label)
        if strcmp(label(j).type , 'DontCare') == 1
             for y = round(label(j).x1):round(label(j).x2)
                 if y == 0
                     y = 1
                 end
                 for x = round(label(j).y1):round(label(j).y2)
                     if x == 0
                         x = 1
                     end
                     img(x,y,:)=mean_image;
                 end
             end
             arr(end+1) = j;
        end
    end
    if not(isempty(arr))
        for a=length(arr):-1:1
             label(arr(a)) = [];
        end
    end
    imwrite(img,fullfile(img_save,sprintf('%06d.png',i)));
    
   %writeLabels(label,test_dir,i);
   fprintf('Test label file %s written!\r\n',file);
end


