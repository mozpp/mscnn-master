
GLOG_logtostderr=1 ../../../build/tools/caffe train \
  --solver=solver_1st.prototxt \
  --weights=mscnn_adapt.caffemodel \
  --gpu=0  2>&1 | tee log_1st.txt

GLOG_logtostderr=1 ../../../build/tools/caffe train \
  --solver=solver_2nd.prototxt \
  --weights=mscnn_kitti_trainval_1st_iter_15000.caffemodel \
  --gpu=0  2>&1 | tee log_2nd.txt

GLOG_logtostderr=1 ../../../build/tools/caffe train \
  --solver=solver_1st_cont.prototxt \
  --weights=mscnn_adapt.caffemodel \
  --gpu=0  2>&1 | tee log_1st_cont.txt

GLOG_logtostderr=1 ../../../build/tools/caffe train \
  --solver=solver_2nd_cont.prototxt \
  --weights=mscnn_1st_cont_iter_15000.caffemodel \
  --gpu=0  2>&1 | tee log_2nd_cont.txt
