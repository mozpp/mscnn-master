GLOG_logtostderr=1 ../../../build/tools/caffe train \
  --solver=solver_2nd.prototxt \
  --weights=mscnn_kitti_trainval_1st_iter_15000.caffemodel \
  --gpu=0
