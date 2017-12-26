#include <algorithm>
#include <vector>

#include "caffe/layers/tracking_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void TrackingLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),  // a
      bottom[1]->gpu_data(),  // b
      diff_relative_.mutable_gpu_data());  // a_i-b_i
  caffe_gpu_powx(
      count,
      diff_relative_.mutable_gpu_data(),  // a_i-b_i
      Dtype(2),
      diff_relative_sq_.mutable_gpu_data());  // (a_i-b_i)^2
  caffe_gpu_gemv(
      CblasNoTrans,
      bottom[0]->num(),
      bottom[0]->channels(),
      Dtype(1.0),
      diff_relative_sq_.gpu_data(),  // (a_i-b_i)^2
      summer_vec_relative_.gpu_data(),
      Dtype(0.0),
      dist_relative_sq_.mutable_gpu_data());  // \Sum (a_i-b_i)^2
  Dtype margin = this->layer_param_.contrastive_loss_param().margin();
  bool legacy_version =
      this->layer_param_.contrastive_loss_param().legacy_version();
  Dtype loss(0.0);
  for (int i = 0; i < bottom[0]->num(); ++i) {
    if (static_cast<int>(bottom[2]->cpu_data()[i])) {  // similar pairs
      loss += dist_relative_sq_.cpu_data()[i];
    } else {  // dissimilar pairs
      loss += std::max(margin - diff_relative_.cpu_data()[i], Dtype(0.0));// change by YangXS: dist_relative_sq_ to diff_relative_
    }
  }
  loss = loss / static_cast<Dtype>(bottom[0]->num()) / Dtype(2);
  
  //change by GaoJY, adding euclidean loss///////////////////////////////////////////////
  int count_detect = bottom[3]->count();
  Dtype lambda1 = this->layer_param_.tracking_loss_param().lambda1();
  caffe_gpu_sub(
      count_detect,
      bottom[3]->gpu_data(),
      bottom[5]->gpu_data(),
      diff_dectect_.mutable_gpu_data());
  Dtype dot;
  caffe_gpu_dot(count_detect, diff_dectect_.gpu_data(), diff_dectect_.gpu_data(), &dot);
  
  int count_detect_1 = bottom[4]->count();
  caffe_gpu_sub(
      count_detect_1,
      bottom[4]->gpu_data(),
      bottom[6]->gpu_data(),
      diff_dectect_1_.mutable_gpu_data());
  Dtype dot1;
  caffe_gpu_dot(count_detect_1, diff_dectect_1_.gpu_data(), diff_dectect_1_.gpu_data(), &dot1);
  
  loss = loss + lambda1 * (dot + dot1) / static_cast<Dtype>(bottom[0]->num()) / Dtype(2);
  //change by GaoJY, adding euclidean loss end///////////////////////////////////////////////
  
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
__global__ void CLLBackward(const int count, const int channels,
    const Dtype margin, const bool legacy_version, const Dtype alpha,
    const Dtype* y, const Dtype* diff_relative_, const Dtype* dist_relative_sq_,
    Dtype *bottom_diff) {
  CUDA_KERNEL_LOOP(i, count) {
    int n = i / channels;  // the num index, to access y and dist_sq
    if (static_cast<int>(y[n])) {  // similar pairs
      bottom_diff[i] = alpha * diff_relative_[i];
    } else {  // dissimilar pairs
      // change by YangXS:
      if ((margin - diff_relative_[n]) > 0.0) {
        bottom_diff[i] = -1/2.0 * alpha;
      } else {
        bottom_diff[i] = 0;
      }
    }
  }
}

template <typename Dtype>
void TrackingLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const int count = bottom[0]->count();
      const int channels = bottom[0]->channels();
      Dtype margin = this->layer_param_.contrastive_loss_param().margin();
      const bool legacy_version =
          this->layer_param_.contrastive_loss_param().legacy_version();
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha_relative = sign * top[0]->cpu_diff()[0] /
          static_cast<Dtype>(bottom[0]->num());
      // NOLINT_NEXT_LINE(whitespace/operators)
      CLLBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, channels, margin, legacy_version, alpha_relative,
          bottom[2]->gpu_data(),  // pair similarity 0 or 1
          diff_relative_.gpu_data(),  // the cached eltwise difference between a and b
          dist_relative_sq_.gpu_data(),  // the cached square distance between a and b
          bottom[i]->mutable_gpu_diff());
      CUDA_POST_KERNEL_CHECK;
    }
  }
  
  //change by GaoJY, adding euclidean loss///////////////////////////////////////////////
  Dtype lambda1 = this->layer_param_.tracking_loss_param().lambda1();
  for (int i = 3; i < 5; ++i) {
    if (propagate_down[i] && i == 3) {
      const Dtype alpha = lambda1 * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_gpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_dectect_.gpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_gpu_diff());  // b
    }
	
	if (propagate_down[i] && i == 4) {
      const Dtype alpha_1 = lambda1 * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_gpu_axpby(
          bottom[i]->count(),              // count
          alpha_1,                              // alpha
          diff_dectect_1_.gpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_gpu_diff());  // b
    }
  }
  //change by GaoJY, adding euclidean loss end///////////////////////////////////////////////
}

INSTANTIATE_LAYER_GPU_FUNCS(TrackingLossLayer);

}  // namespace caffe
