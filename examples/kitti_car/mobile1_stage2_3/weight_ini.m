clear all; close all;
addpath('../../../matlab/');
addpath('../../../utils/');
caffe.reset_all();
%caffe.set_mode_cpu();
%model1='../../../models/VGG/vgg_16.prototxt';
model1='../mobile1_stage2_2/trainval_2nd.prototxt';
model2='trainval_2nd.prototxt';

weight1='../mobile1_stage2_2/mscnn_kitti_trainval_2nd_iter_35000.caffemodel';
%weight1='../../../models/VGG/VGG_ILSVRC_16_layers.caffemodel';
weight2='mscnn_kitti_trainval_2nd_iter_19.caffemodel';
net1=caffe.Net(model1,weight1,'train');
net2=caffe.Net(model2,weight2,'train');

net1_fc6_1=net1.layers(['fc6']).params(1).get_data();
net1_fc6_2=net1.layers(['fc6']).params(2).get_data();
prob = net1.blobs('roi_c1').get_data();
fc6_1=net2.layers(['fc6']).params(1).get_data();
fc6_2=net2.layers(['fc6']).params(2).get_data();
size(net1_fc6_1)
size(net1_fc6_2)
size(fc6_1)
size(fc6_2)
% for k0=2:4
% convk_1_1=net1.layers(['conv' num2str(k0) '_1']).params(1).get_data();
% convk_1_2=net1.layers(['conv' num2str(k0) '_1']).params(2).get_data();
% convk_1_new1=net2.layers(['conv' num2str(k0) '_1_new']).params(1).get_data();
% convk_1_new2=net2.layers(['conv' num2str(k0) '_1_new']).params(2).get_data();
% 
% [w h chin chout]=size(convk_1_1);
% convk_1_new1(:,:,1:chin,:)=convk_1_1;
% net2.layers(['conv' num2str(k0) '_1_new']).params(1).set_data(convk_1_new1);
% net2.layers(['conv' num2str(k0) '_1_new']).params(2).set_data(convk_1_2);
% end
% net2.save('mscnn_adapt.caffemodel');
% 
% model3='mscnn_deploy.prototxt';
% weight3='mscnn_adapt.caffemodel';
% net3=caffe.Net(model3,weight3,'train');
% test=net3.layers('conv2_1_new').params(1).get_data();
