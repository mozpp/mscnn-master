% Copyright (c) 2016 The Regents of the University of California
% see mscnn/LICENSE for details
% Written by Zhaowei Cai [zwcai-at-ucsd.edu]
% Please email me if you find bugs, or have suggestions or questions!

clear all; close all;

addpath('../../matlab/');
addpath('../../utils/');

root_dir = 'fcn-8s/';
binary_file = [root_dir 'train_iter_10000.caffemodel'];
assert(exist(binary_file, 'file') ~= 0);
definition_file = [root_dir 'deploy.prototxt'];
assert(exist(definition_file, 'file') ~= 0);
use_gpu = true;
if (~use_gpu)
  caffe.set_mode_cpu();
else
  caffe.set_mode_gpu();  
  gpu_id = 0; caffe.set_device(gpu_id);
end
% Initialize a network
net = caffe.Net(definition_file, binary_file, 'test');

% set KITTI dataset directory
image_dir='/media/b3-542/library2/moz/night_det/dataset/images_val/';
label_dir='/media/b3-542/LIBRARY/moz/dataset/SYSU_Nighttime_Vehicle_Detection_Dataset/MyLabel/';
image_list2=textread('../../data/night_det/ImageSets/val1.txt', '%s');
comp_id = 'kitti_8s_768_35k_test';
image_list = dir([image_dir '*.jpg']);
nImg=length(image_list);

% choose the right input size
% imgW = 1280; imgH = 384;
% imgW = 1920; imgH = 576;
%imgW =1344; imgH = 768;
imgW =704; imgH = 704;

mu = ones(1,1,3); mu(:,:,1:3) = [104 117 123];
mu = repmat(mu,[imgH,imgW,1]);

% bbox de-normalization parameters
bbox_means = [0 0 0 0];
bbox_stds = [0.1 0.1 0.2 0.2];

usedtime=0; 

show = 0; show_thr = 0.5;
% if (show)
%   fig=figure(1); set(fig,'Position',[-50 100 1350 375]);
%   h.axes = axes('position',[0,0,1,1]);
% end

for k = 1:nImg
  
  %test_image = imread([image_dir image_list(k).name],4);
  test_image = imread([image_dir image_list(k).name]);
  %test_image=X;
  if (show)
    %imshow(test_image,'parent',h.axes); axis(h.axes,'image','off'); hold(h.axes,'on');
    imshow(test_image(:,:,1:3));
  end 

  [orgH,orgW,~] = size(test_image);
  ratios = [imgH imgW]./[orgH orgW];
  test_image = imresize(test_image,[imgH imgW]); 
  test_image = single(test_image(:,:,[3 2 1]));
  test_image = bsxfun(@minus,test_image,mu);
  test_image = permute(test_image, [2 1 3]);
  
  % network forward
  tic; 
  %outputs = net.forward({test_image});
  net.forward({test_image});
  outputs = net.blobs('score').get_data();
  pertime=toc;
  usedtime=usedtime+pertime; avgtime=usedtime/k;
  
  exp_score = exp(outputs);
  sum_exp_score = sum(exp_score,3);
  prob = exp_score(:,:,2)./sum_exp_score;
  imshow(prob*255);
  if (show) pause(0.02); end
  if (mod(k,100)==0), fprintf('idx %i/%i, avgtime=%.4fs\n',k,nImg,avgtime); end
end


caffe.reset_all();

