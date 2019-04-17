GLOG_logtostderr=1 ../../../build/tools/caffe train \
  --solver=solver_2nd_additer.prototxt \
  --snapshot=mscnn_kitti_trainval_2nd_iter_25000.solverstate \
  --gpu=0  2>&1 | tee log_4th.txt
