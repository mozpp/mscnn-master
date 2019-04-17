GLOG_logtostderr=1 ../../../build/tools/caffe train \
  --solver=solver_2nd.prototxt \
  --gpu=0  2>&1 | tee log_2nd.txt
