#include <algorithm>
#include <vector>

#include "caffe/layers/tracking_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void TrackingLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  CHECK_EQ(bottom[2]->channels(), 1);
  CHECK_EQ(bottom[2]->height(), 1);
  CHECK_EQ(bottom[2]->width(), 1);
  diff_relative_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  diff_relative_sq_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  dist_relative_sq_.Reshape(bottom[0]->num(), 1, 1, 1);
  // vector of ones used to sum along channels
  summer_vec_relative_.Reshape(bottom[0]->channels(), 1, 1, 1);
  for (int i = 0; i < bottom[0]->channels(); ++i)
    summer_vec_relative_.mutable_cpu_data()[i] = Dtype(1);

//change by GaoJY, adding euclidean loss///////////////////////////////////////////////
  CHECK_EQ(bottom[3]->num(), bottom[5]->num())
      << "The data and label should have the same number.";
  vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
  top[0]->Reshape(loss_shape);
  CHECK_EQ(bottom[3]->count(1), bottom[5]->count(1))
      << "Inputs must have the same dimension.";
  diff_dectect_.ReshapeLike(*bottom[3]);
  
  CHECK_EQ(bottom[4]->num(), bottom[6]->num())
      << "The data and label should have the same number.";
  CHECK_EQ(bottom[4]->count(1), bottom[6]->count(1))
      << "Inputs must have the same dimension.";
  diff_dectect_1_.ReshapeLike(*bottom[4]);
//change by GaoJY, adding euclidean loss///////////////////////////////////////////////
}


//////////////////////////////////////////// noted and changed by YangXS and GaoJY///////////////////////////////////////////////

template <typename Dtype>
void TrackingLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),  // a
      bottom[1]->cpu_data(),  // b
      diff_relative_.mutable_cpu_data());  // a_i-b_i
  const int channels = bottom[0]->channels();
  Dtype margin = this->layer_param_.tracking_loss_param().margin();
  bool legacy_version =
      this->layer_param_.tracking_loss_param().legacy_version();// will be unused
  Dtype loss(0.0);
  for (int i = 0; i < bottom[0]->num(); ++i) {
    dist_relative_sq_.mutable_cpu_data()[i] = caffe_cpu_dot(channels,
        diff_relative_.cpu_data() + (i*channels), diff_relative_.cpu_data() + (i*channels));
    if (static_cast<int>(bottom[2]->cpu_data()[i])) {  // similar pairs
      loss += dist_relative_sq_.cpu_data()[i];
    } else {  // dissimilar pairs
      //change by YangXS
      loss += std::max(margin - diff_relative_.cpu_data()[i], Dtype(0.0));
    }
  }
  loss = loss / static_cast<Dtype>(bottom[0]->num()) / Dtype(2);
  
  //change by GaoJY, adding euclidean loss///////////////////////////////////////////////
  Dtype lambda1 = this->layer_param_.tracking_loss_param().lambda1();
  int count_detect = bottom[3]->count();
  caffe_sub(
      count_detect,
      bottom[3]->cpu_data(),
      bottom[5]->cpu_data(),
      diff_dectect_.mutable_cpu_data());
	  
  int count_detect_1 = bottom[4]->count();
  caffe_sub(
      count_detect_1,
      bottom[4]->cpu_data(),
      bottom[6]->cpu_data(),
      diff_dectect_1_.mutable_cpu_data());
  Dtype dot = caffe_cpu_dot(count_detect, diff_dectect_.cpu_data(), diff_dectect_.cpu_data());
  Dtype dot1 = caffe_cpu_dot(count_detect_1, diff_dectect_1_.cpu_data(), diff_dectect_1_.cpu_data());
  dot = lambda1 * (dot + dot1);
  loss = loss + dot / static_cast<Dtype>(bottom[0]->num()) / Dtype(2);
  //change by GaoJY, adding euclidean loss end///////////////////////////////////////////////
  
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void TrackingLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype margin = this->layer_param_.tracking_loss_param().margin();
  bool legacy_version =
      this->layer_param_.tracking_loss_param().legacy_version();// will be unused
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] /
          static_cast<Dtype>(bottom[i]->num());
      int num = bottom[i]->num();
      int channels = bottom[i]->channels();
      for (int j = 0; j < num; ++j) {
        Dtype* bout = bottom[i]->mutable_cpu_diff();
        if (static_cast<int>(bottom[2]->cpu_data()[j])) {  // similar pairs
          caffe_cpu_axpby(
              channels,
              alpha,
              diff_relative_.cpu_data() + (j*channels),
              Dtype(0.0),
              bout + (j*channels));
        } else {  // dissimilar pairs
          // changed by YangXS
          if ((margin - diff_relative_.cpu_data()[j]) > Dtype(0.0)) {
            const Dtype tmp_one = Dtype(1.0);
            caffe_cpu_axpby(
                channels,
                -1/(Dtype)2.0*alpha,
                &tmp_one,
                Dtype(0.0),
                bout + (j*channels));
          } else {
            caffe_set(channels, Dtype(0), bout + (j*channels));
          }
        }
      }
    }
  }
  
  //change by GaoJY, adding euclidean loss///////////////////////////////////////////////

  Dtype lambda1 = this->layer_param_.tracking_loss_param().lambda1();
  for (int i = 3; i < 5; ++i) {
	  if (propagate_down[i] && i == 3) {
		  const Dtype alpha_detect = lambda1 * top[0]->cpu_diff()[0] / bottom[i]->num();
		  caffe_cpu_axpby(                 //Y=alpha * X +beta*Y 
		  bottom[i]->count(),              // count
		  alpha_detect,                              // alpha
		  diff_dectect_.cpu_data(),                   // a
		  Dtype(0),                           // beta
		  bottom[i]->mutable_cpu_diff());  // b
	  }
	  
	  if (propagate_down[i] && i == 4) {
		  const Dtype alpha_detect_1 = lambda1 * top[0]->cpu_diff()[0] / bottom[i]->num();
		  caffe_cpu_axpby(                 //Y=alpha * X +beta*Y 
		  bottom[i]->count(),              // count
		  alpha_detect_1,                              // alpha
		  diff_dectect_1_.cpu_data(),                   // a
		  Dtype(0),                           // beta
		  bottom[i]->mutable_cpu_diff());  // b
	  }
  }
  
  //change by GaoJY, adding euclidean loss///////////////////////////////////////////////
}

#ifdef CPU_ONLY
STUB_GPU(TrackingLossLayer);
#endif

INSTANTIATE_CLASS(TrackingLossLayer);
REGISTER_LAYER_CLASS(TrackingLoss);

}  // namespace caffe
