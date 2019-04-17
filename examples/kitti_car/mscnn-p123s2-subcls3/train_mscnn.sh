GLOG_logtostderr=1 ../../../build/tools/caffe train \
  --solver=solver_1st_sub.prototxt \
  --weights=mscnn_adapt.caffemodel \
  --gpu=0  2>&1 | tee log_1st.txt

GLOG_logtostderr=1 ../../../build/tools/caffe train \
  --solver=solver_2nd_sub.prototxt \
  --weights=mscnn_kitti_trainval_1st_sub_iter_15000.caffemodel \
  --gpu=0  2>&1 | tee log_2nd.txt

GLOG_logtostderr=1 ../../../build/tools/caffe train \
  --solver=solver_2nd_fix.prototxt \
  --weights=mscnn_kitti_trainval_2nd_sub_iter_35000.caffemodel \
  --gpu=0  2>&1 | tee log_3rd.txt

GLOG_logtostderr=1 ../../../build/tools/caffe train \
  --solver=solver_2nd.prototxt \
  --weights=mscnn_kitti_trainval_2nd_fix_iter_10000.caffemodel \
  --gpu=0  2>&1 | tee log_4th.txt
