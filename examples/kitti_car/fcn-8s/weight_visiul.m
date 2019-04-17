clear all; close all;
addpath('../../../matlab/');
addpath('../../../utils/');
caffe.reset_all();
%caffe.set_mode_cpu();
%model1='../../../models/VGG/vgg_16.prototxt';
model2='train_val.prototxt';

%weight1='../../../models/VGG/VGG_ILSVRC_16_layers.caffemodel';
weight2='train_iter_80000.caffemodel';
net=caffe.Net(model2,weight2,'train');

net.forward({});
outputs = net.blobs('data123').get_data();
ch4 = net.blobs('ch4').get_data();
figure
subplot(1,2,1);
image(uint8(outputs(:,:,:)));
subplot(1,2,2);
image(uint8(ch4*128));

% image_dir='/media/b3-542/library2/moz/night_det/result/concat/blob_jan_xgt_1ov255_train/';
% image_list = dir([image_dir '*.png']);
% for k = 1:1
%   
%   %test_image = imread([image_dir image_list(k).name],4);
%   [X,map,alpha] = imread([image_dir image_list(k).name]);
%   test_image=cat(3,X,alpha);
%   imshow(alpha*128);
% end