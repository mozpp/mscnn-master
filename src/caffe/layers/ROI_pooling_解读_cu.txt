namespace caffe {

  template <typename Dtype>
  __global__ void ROIPoolForward(const int nthreads, //线程数
                                 const Dtype* bottom_data,// 传入层数据指针
                                 const Dtype spatial_scale,//与原始图像相比,传入特征图的下采样倍数
                                 const int channels, //通道数
                                 const int height, //传入特征图高度
                                 const int width, //传入特征图宽度
                                 const int pooled_height, //传出特征图高度
                                 const int pooled_width, //传出特征图宽度
                                 const Dtype* bottom_rois, //传入层对应的ROI区域
                                 Dtype* top_data, //传出层数据指针
                                 int* argmax_data) //区域最大值位置索引
  {
    CUDA_KERNEL_LOOP(index, nthreads) {
      int pw = index % pooled_width; //表示传出数据对应的宽度索引
      int ph = (index / pooled_width) % pooled_height;//表示传出数据对应的高度索引
      int c = (index / pooled_width / pooled_height) % channels;//表示传出数据对应的通道索引
      int n = index / pooled_width / pooled_height / channels;//表示传出数据对应的batch索引
      //获取当前ROI在feature map上的位置
      bottom_rois += n * 5;
      int roi_batch_ind = bottom_rois[0];
      int roi_start_w = round(bottom_rois[1] * spatial_scale);
      int roi_start_h = round(bottom_rois[2] * spatial_scale);
      int roi_end_w = round(bottom_rois[3] * spatial_scale);
      int roi_end_h = round(bottom_rois[4] * spatial_scale);
      // 根据需要输出的feature map的大小,计算输入ROI中bin尺寸
      // 此时还是浮点数
      int roi_width = max(roi_end_w - roi_start_w + 1, 1);
      int roi_height = max(roi_end_h - roi_start_h + 1, 1);

      Dtype bin_size_h = static_cast<Dtype>(roi_height)
        / static_cast<Dtype>(pooled_height);
      Dtype bin_size_w = static_cast<Dtype>(roi_width)
        / static_cast<Dtype>(pooled_width);
      //获取bin的起始结束坐标,并做保护措施,注意起始坐标获取采用floor函数
      //即向下取整,结束坐标获取采用ceil函数,即向上取整,即会使得相邻两个bin有
      //重叠
      int hstart = static_cast<int>(floor(static_cast<Dtype>(ph)
        * bin_size_h));
      int wstart = static_cast<int>(floor(static_cast<Dtype>(pw)
        * bin_size_w));
      int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1)
        * bin_size_h));
      int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1)
        * bin_size_w));
      // (Add roi offsets and clip to input boundaries)
      hstart = min(max(hstart + roi_start_h, 0), height);
      hend = min(max(hend + roi_start_h, 0), height);
      wstart = min(max(wstart + roi_start_w, 0), width);
      wend = min(max(wend + roi_start_w, 0), width);
      bool is_empty = (hend <= hstart) || (wend <= wstart);

      // maxval用于记录每一个小块的最大值
      Dtype maxval = is_empty ? 0 : -FLT_MAX;
      // (If nothing is pooled, argmax = -1 causes nothing to be backprop'd)
      // maxidx 用于记录当前bin的最大值索引位置,默认值-1
      int maxidx = -1;
      // 根据传入参数将bottom_data偏移到感兴趣的那一层
      bottom_data += (roi_batch_ind * channels + c) * height * width;
      // 操作感兴趣层对应bin的数据,获取最大值,并记录最大值位置
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          int bottom_index = h * width + w;
          if (bottom_data[bottom_index] > maxval) {
            maxval = bottom_data[bottom_index];
            maxidx = bottom_index;
          }
        }
      }
      top_data[index] = maxval;
      argmax_data[index] = maxidx;
    }
  }

  template <typename Dtype>
  void ROIPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* bottom_rois = bottom[1]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    int* argmax_data = max_idx_.mutable_gpu_data();
    int count = top[0]->count();//前向运算的count数是由传出feature map的尺寸决定
    // NOLINT_NEXT_LINE(whitespace/operators)
    ROIPoolForward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
      count, bottom_data, spatial_scale_, channels_, height_, width_,
      pooled_height_, pooled_width_, bottom_rois, top_data, argmax_data);
    CUDA_POST_KERNEL_CHECK;
  }

  template <typename Dtype>
  __global__ void ROIPoolBackward(const int nthreads, //线程数
                                  const Dtype* top_diff, //传出层特征图梯度值
                                  const int* argmax_data, //区域最大值位置索引
                                  const int num_rois, //roi的个数
                                  const Dtype spatial_scale, //与原始图像相比,传入特征图的下采样倍数
                                  const int channels, //传入层通道数
                                  const int height, //传入层特征图高度
                                  const int width, //传入层特征图宽度
                                  const int pooled_height, //传出特征图高度
                                  const int pooled_width, //传出特征图宽度
                                  Dtype* bottom_diff, //传入层特征图梯度值
                                  const Dtype* bottom_rois) {//传入层ROI数据
    CUDA_KERNEL_LOOP(index, nthreads) {
      // (n, c, h, w) coords in bottom data
      int w = index % width;//表示传入数据对应的宽度索引
      int h = (index / width) % height;//表示传入数据对应的高度索引
      int c = (index / width / height) % channels;//表示传入数据对应的通道数索引
      int n = index / width / height / channels;//表示传入数据对应的batch索引

      Dtype gradient = 0;
      // Accumulate gradient over all ROIs that pooled this element
      // 累加所有ROI中跟这一点有关的梯度值
      for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
        const Dtype* offset_bottom_rois = bottom_rois + roi_n * 5;
        int roi_batch_ind = offset_bottom_rois[0];
        // Skip if ROI's batch index doesn't match n
        if (n != roi_batch_ind) {
          continue;
        }

        int roi_start_w = round(offset_bottom_rois[1] * spatial_scale);
        int roi_start_h = round(offset_bottom_rois[2] * spatial_scale);
        int roi_end_w = round(offset_bottom_rois[3] * spatial_scale);
        int roi_end_h = round(offset_bottom_rois[4] * spatial_scale);

        // Skip if ROI doesn't include (h, w)
        //如果此点不在当前ROI内,则继续循环,若所有ROI都不包含此点,则此点梯度值为0
        const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
          h >= roi_start_h && h <= roi_end_h);
        if (!in_roi) {
          continue;
        }

        int offset = (roi_n * channels + c) * pooled_height * pooled_width;
        const Dtype* offset_top_diff = top_diff + offset;
        const int* offset_argmax_data = argmax_data + offset;

        // Compute feasible set of pooled units that could have pooled
        // this bottom unit

        // Force malformed ROIs to be 1x1

        // 以下操作是根据当前传入层数据的坐标查找传出层中对应位置的数据,并记录填充
        // 若当前传入层数据点对一个以上ROI做了贡献,则梯度累加
        int roi_width = max(roi_end_w - roi_start_w + 1, 1);
        int roi_height = max(roi_end_h - roi_start_h + 1, 1);
        Dtype bin_size_h = static_cast<Dtype>(roi_height)
          / static_cast<Dtype>(pooled_height);
        Dtype bin_size_w = static_cast<Dtype>(roi_width)
          / static_cast<Dtype>(pooled_width);

        int phstart = floor(static_cast<Dtype>(h - roi_start_h) / bin_size_h);
        int phend = ceil(static_cast<Dtype>(h - roi_start_h + 1) / bin_size_h);
        int pwstart = floor(static_cast<Dtype>(w - roi_start_w) / bin_size_w);
        int pwend = ceil(static_cast<Dtype>(w - roi_start_w + 1) / bin_size_w);

        phstart = min(max(phstart, 0), pooled_height);
        phend = min(max(phend, 0), pooled_height);
        pwstart = min(max(pwstart, 0), pooled_width);
        pwend = min(max(pwend, 0), pooled_width);

        for (int ph = phstart; ph < phend; ++ph) {
          for (int pw = pwstart; pw < pwend; ++pw) {
            if (offset_argmax_data[ph * pooled_width + pw] == (h * width + w)) {
              gradient += offset_top_diff[ph * pooled_width + pw];
            }
          }
        }
      }
      bottom_diff[index] = gradient;
    }
  }

  template <typename Dtype>
  void ROIPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (!propagate_down[0]) {
      return;
    }
    // bottom和top的相对关系由forward决定
    const Dtype* bottom_rois = bottom[1]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();//前向运算的count数是由传入feature map的尺寸决定
    caffe_gpu_set(count, Dtype(0.), bottom_diff);
    const int* argmax_data = max_idx_.gpu_data();
    // NOLINT_NEXT_LINE(whitespace/operators)
    ROIPoolBackward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
      count, top_diff, argmax_data, top[0]->num(), spatial_scale_, channels_,
      height_, width_, pooled_height_, pooled_width_, bottom_diff, bottom_rois);
    CUDA_POST_KERNEL_CHECK;
  }

  INSTANTIATE_LAYER_GPU_FUNCS(ROIPoolingLayer);

}  // namespace caffe
