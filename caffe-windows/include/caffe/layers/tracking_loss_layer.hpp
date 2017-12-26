#ifndef CAFFE_TRACKING_LOSS_LAYER_HPP_
#define CAFFE_TRACKING_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {


template <typename Dtype>
class TrackingLossLayer : public LossLayer<Dtype> {
 public:
  explicit TrackingLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_relative_(), diff_dectect_() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int ExactNumBottomBlobs() const { return 7; }
  virtual inline const char* type() const { return "TrackingLoss"; }
 
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return ((bottom_index != 2) && (bottom_index != 5) && (bottom_index != 6));//changed by GaoJY, 
  }

 protected:
  /// @copydoc TrackingLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

 
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> diff_relative_;  // cached for backward pass,relative
  Blob<Dtype> dist_relative_sq_;  // cached for backward pass,relative
  Blob<Dtype> diff_relative_sq_;  // tmp storage for gpu forward pass,relative
  Blob<Dtype> summer_vec_relative_;  // tmp storage for gpu forward pass,relative
  Blob<Dtype> diff_dectect_;//cached for euclidean loss backward pass
  Blob<Dtype> diff_dectect_1_;//cached for euclidean loss backward pass
};

}  // namespace caffe

#endif  // CAFFE_CONTRASTIVE_LOSS_LAYER_HPP_