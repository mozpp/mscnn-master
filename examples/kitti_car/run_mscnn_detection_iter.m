% Copyright (c) 2016 The Regents of the University of California
% see mscnn/LICENSE for details
% Written by Zhaowei Cai [zwcai-at-ucsd.edu]
% Please email me if you find bugs, or have suggestions or questions!

clear all; close all;

addpath('../../matlab/');
addpath('../../utils/');
caffe.reset_all();
model_root_dir = 'mscnn-p123s2-subcls3/';
binary_list=dir([model_root_dir 'iter/' '*.caffemodel']);
%root_dir='/media/b3-542/LIBRARY/moz/result/concat/subcls_blob/';
%root_dir='/media/b3-542/LIBRARY/moz/dataset/HK_dataset/concat/';
% img_root_list = {'add_fn0.2_val','add_fn0.5_val','add_fp0.2_val','add_fp0.3_val',...
%     'add_fp0.5_val','rec0.5pre0.5_val','rec0.6pre0.6_val','rec0.7pre0.7_val',...
%     'rec0.8pre0.8_val'};
root_dir='/media/b3-542/library2/moz/night_det/result/concat/blob_jan_xgt_1ov255_train/';
img_root_list={'subcls_blob_train'};
% img_root_list={'blob_clsf_2Ch_val'};%'subcls_blob_val'
nCaffemodel=length(binary_list);
nVal=length(img_root_list);
for k0=1:nCaffemodel
for k1=1:nVal
img_dir1=img_root_list{1,k1};
binary_file = [model_root_dir 'iter/' binary_list(k0).name];
assert(exist(binary_file, 'file') ~= 0);
definition_file = [model_root_dir 'mscnn_deploy.prototxt'];
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
image_dir = [root_dir img_dir1 '/'];
label_dir='/media/b3-542/LIBRARY/moz/dataset/SYSU_Nighttime_Vehicle_Detection_Dataset/MyLabel/';
image_list2=textread('../../data/night_det/ImageSets/val1.txt', '%s');
comp_id = [binary_list(k0).name '_' img_dir1];
%image_list = dir([image_dir '*.jpg']);
image_list = dir([image_dir '*.png']);
nImg=length(image_list);

% choose the right input size
% imgW = 1280; imgH = 384;
% imgW = 1920; imgH = 576;
imgW =1344; imgH = 768;

mu = ones(1,1,4); mu(:,:,1:4) = [104 117 123 0];
mu = repmat(mu,[imgH,imgW,1]);

% bbox de-normalization parameters
bbox_means = [0 0 0 0];
bbox_stds = [0.1 0.1 0.2 0.2];

% non-maxisum suppression parameters
pNms.type = 'maxg'; pNms.overlap = 0.5; pNms.ovrDnm = 'union';%pNms.type = 'maxg'/'maxg_my1'/'maxg_my2'

cls_ids = [2]; num_cls=length(cls_ids); 
obj_names = {'bg','car','car0light','truck','tram'};
final_detect_boxes = cell(nImg,num_cls); final_proposals = cell(nImg,1);
proposal_thr = -10; usedtime=0; 

show = 0; show_thr = 0.5;
% if (show)
%   fig=figure(1); set(fig,'Position',[-50 100 1350 375]);
%   h.axes = axes('position',[0,0,1,1]);
% end

for k = 1:nImg
  
  %test_image = imread([image_dir image_list(k).name],4);
  [X,map,alpha] = imread([image_dir image_list(k).name]);
  test_image=cat(3,X,alpha);
  %test_image=X;
  if (show)
    %imshow(test_image,'parent',h.axes); axis(h.axes,'image','off'); hold(h.axes,'on');
    imshow(test_image(:,:,1:3));
  end
  
%   clear tline1;
%   %[label_dir image_list2{k} '.txt']
%   ffid = fopen([label_dir image_list2{k} '.txt'],'r');
%   tline = fgetl(ffid);
%   i = 1;
%   while feof(ffid) == 0
%       tline1{i,1} = fgetl(ffid);
%       i = i+1;
%   end
%   if i==1
%       tline1{1,1}=tline;
%   else
%       tline1=[tline;tline1];
%   end
%   A=zeros(0,4);
% for k_1=1:length(tline1)
% 
%     str=strsplit(tline1{k_1,1});
%     w=str2num(str{1,7})-str2num(str{1,5});
%     h=str2num(str{1,8})-str2num(str{1,6});
%     str=[str2num(str{1,5}),str2num(str{1,6}),w,h];
%     %str=[str2num(str{1,5}),str2num(str{1,6}),str2num(str{1,7}),str2num(str{1,8})];
%     A=[A;str];
% end  
%    fclose(ffid);  

  [orgH,orgW,~] = size(test_image);
  ratios = [imgH imgW]./[orgH orgW];
  test_image = imresize(test_image,[imgH imgW]); 
  test_image = single(test_image(:,:,[3 2 1 4]));
  test_image = bsxfun(@minus,test_image,mu);
  test_image = permute(test_image, [2 1 3]);

  % network forward
  tic; outputs = net.forward({test_image}); pertime=toc;
  usedtime=usedtime+pertime; avgtime=usedtime/k;
  
  tmp=squeeze(outputs{1}); bbox_preds = tmp';
  tmp=squeeze(outputs{2}); cls_pred = tmp'; 
  tmp=squeeze(outputs{3}); tmp = tmp'; tmp = tmp(:,2:end); 
  tmp(:,3) = tmp(:,3)-tmp(:,1); tmp(:,4) = tmp(:,4)-tmp(:,2); 
  proposal_pred = tmp; proposal_score = proposal_pred(:,end);
  
  % filtering some bad proposals
  keep_id = find(proposal_score>=proposal_thr & proposal_pred(:,3)~=0 & proposal_pred(:,4)~=0);
  proposal_pred = proposal_pred(keep_id,:); 
  bbox_preds = bbox_preds(keep_id,:); cls_pred = cls_pred(keep_id,:);
    
  proposals = double(proposal_pred);
  proposals(:,1) = proposals(:,1)./ratios(2); 
  proposals(:,3) = proposals(:,3)./ratios(2);
  proposals(:,2) = proposals(:,2)./ratios(1);
  proposals(:,4) = proposals(:,4)./ratios(1);
  final_proposals{k} = proposals;

  for i = 1:num_cls
    id = cls_ids(i); bbset = [];
    bbox_pred = bbox_preds(:,id*4-3:id*4); 

    % bbox de-normalization
    bbox_pred = bbox_pred.*repmat(bbox_stds,[size(bbox_pred,1) 1]);
    bbox_pred = bbox_pred+repmat(bbox_means,[size(bbox_pred,1) 1]);

    exp_score = exp(cls_pred);
    sum_exp_score = sum(exp_score,2);
    prob = exp_score(:,id)./sum_exp_score; 
    ctr_x = proposal_pred(:,1)+0.5*proposal_pred(:,3);
    ctr_y = proposal_pred(:,2)+0.5*proposal_pred(:,4);
    tx = bbox_pred(:,1).*proposal_pred(:,3)+ctr_x;
    ty = bbox_pred(:,2).*proposal_pred(:,4)+ctr_y;
    tw = proposal_pred(:,3).*exp(bbox_pred(:,3));
    th = proposal_pred(:,4).*exp(bbox_pred(:,4));
    tx = tx-tw/2; ty = ty-th/2;
    tx = tx./ratios(2); tw = tw./ratios(2);
    ty = ty./ratios(1); th = th./ratios(1);

    % clipping bbs to image boarders
    tx = max(0,tx); ty = max(0,ty);
    tw = min(tw,orgW-tx); th = min(th,orgH-ty);     
    bbset = double([tx ty tw th prob]);
    bbset_my=bbset;
%     bbset_my2=zeros(0,5);
%     [m1 n1]=size(bbset_my);
%     for i1=1:m1
%         if bbset_my(i1,5)>0.7
%             bbset_my2=[bbset_my2;bbset_my(i1,:)];
%         end
%     end
%     bbset=bbset_my2;
    
    idlist = 1:size(bbset,1); bbset = [bbset idlist'];
    bbset=bbNms(bbset,pNms);
    %bbset=bbNms_my(bbset,pNms);
    final_detect_boxes{k,i} = bbset(:,1:5);
    
    if (show) 
      proposals_show = zeros(0,5); bbs_show = zeros(0,6);
      if (size(bbset,1)>0) 
        show_id = find(bbset(:,5)>=show_thr);
        bbs_show = bbset(show_id,:);
        proposals_show = proposals(bbs_show(:,6),:); 
      end
      % proposal
%       for j = 1:size(proposals_show,1)
%         rectangle('Position',proposals_show(j,1:4),'EdgeColor','g','LineWidth',2);
%         show_text = sprintf('%.2f',proposals_show(j,5));
%         x = proposals_show(j,1)+0.5*proposals_show(j,3);
%         text(x,proposals_show(j,2),show_text,'color','r', 'BackgroundColor','k','HorizontalAlignment',...
%             'center', 'VerticalAlignment','bottom','FontWeight','bold', 'FontSize',8);
%       end 
      % detection
      for j = 1:size(bbs_show,1)
        rectangle('Position',bbs_show(j,1:4),'EdgeColor','y','LineWidth',1.5);
        show_text = sprintf('%s=%.2f',obj_names{id},bbs_show(j,5));
        x = bbs_show(j,1)+0.5*bbs_show(j,3);
        text(x,bbs_show(j,2),show_text,'color','r', 'BackgroundColor','k','HorizontalAlignment',...
            'center', 'VerticalAlignment','bottom','FontWeight','bold', 'FontSize',8);
      end
      %ground true
%       for j = 1:size(A,1)
%           rectangle('Position',A(j,1:4),'EdgeColor','r','LineWidth',1.5);
%       end
      %% if uncomment the following three lines, the results can be saved in 'results/'
       handle=gcf;%gcf
       saveas(handle,['results/'  num2str(k) '.jpg']);
       %saveas(handle,['results/'  image_list(k).name '.jpg']);
%       %saveas(handle,['results/'  image_list{k,1} '.jpg']);
       clear handle;
    end
  end
  if (show) pause(0.02); end
  if (mod(k,100)==0), fprintf('idx %i/%i, avgtime=%.4fs\n',k,nImg,avgtime); end
end

for i=1:nImg
  for j=1:num_cls
    final_detect_boxes{i,j}=[ones(size(final_detect_boxes{i,j},1),1)*i final_detect_boxes{i,j}]; 
  end
  final_proposals{i}=[ones(size(final_proposals{i},1),1)*i final_proposals{i}];
end
for j=1:num_cls
  id = cls_ids(j);
  save_detect_boxes=cell2mat(final_detect_boxes(:,j));
  dlmwrite(['detections/' comp_id '_' obj_names{id} '.txt'],save_detect_boxes);
end
final_proposals=cell2mat(final_proposals);
%dlmwrite(['proposals/' comp_id '.txt'],final_proposals);

caffe.reset_all();
end
end