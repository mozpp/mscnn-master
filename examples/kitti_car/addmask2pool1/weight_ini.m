clear all; close all;
addpath('../../../matlab/');
addpath('../../../utils/');
caffe.reset_all();
%caffe.set_mode_cpu();
model1='../../../models/VGG/vgg_16.prototxt';
model2='mscnn_deploy.prototxt';

weight1='../../../models/VGG/VGG_ILSVRC_16_layers.caffemodel';
weight2='mscnn_kitti_trainval_0_iter_0.caffemodel';
net1=caffe.Net(model1,weight1,'train');
net2=caffe.Net(model2,weight2,'train');

conv2_1_1=net1.layers('conv2_1').params(1).get_data();
conv2_1_2=net1.layers('conv2_1').params(2).get_data();
conv2_1_new1=net2.layers('conv2_1_new').params(1).get_data();
conv2_1_new2=net2.layers('conv2_1_new').params(2).get_data();

conv2_1_new1(:,:,1:64,:)=conv2_1_1;
net2.layers('conv2_1_new').params(1).set_data(conv2_1_new1);
net2.layers('conv2_1_new').params(2).set_data(conv2_1_2);

net2.save('mscnn_adapt.caffemodel');

model3='mscnn_deploy.prototxt';
weight3='mscnn_adapt.caffemodel';
net3=caffe.Net(model3,weight3,'train');
test=net3.layers('conv2_1_new').params(1).get_data();
