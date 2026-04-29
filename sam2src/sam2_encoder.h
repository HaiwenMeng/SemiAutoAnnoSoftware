/**
  @Author: mpj
  @Date  : 2026/4/6 17:12
  @version V1.0
  @since C++17
**/

#ifndef SEGMENT_ANYTHING2_SAM2_ENCODER_H
#define SEGMENT_ANYTHING2_SAM2_ENCODER_H

#include <numeric>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include "infer.h"

namespace Sam2 {
  struct ImageEmbedding {
    float* image_embedding_;
    float* high_res_feats0_;
    float* high_res_feats1_;
  };

  class Encoder {
  public:
    explicit Encoder(const std::string& model_path, int cuda_device_id = 0);

    ~Encoder();

    // 返回是gpu上地址
    ImageEmbedding forward(const cv::Mat& bgr_image);

    // 返回是gpu上地址
    [[nodiscard]] ImageEmbedding get_image_embedding() const;

    [[nodiscard]] bool is_image_embedding_ready() const { return is_image_embedding_ready_; }

    bool reset_image_embedding();

  private:
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<int> m_input_dims;
    std::vector<int> m_image_embed_dims;
    std::vector<int> m_high_res_feats0_dims;
    std::vector<int> m_high_res_feats1_dims;

    std::shared_ptr<trt::Infer> m_trt;
    // 预处理缓存，用来存储预处理之后的图片，放在gpu上
    trt::Memory<float> m_input_buffer;
    // 输出缓存，用来存储输出结果，放在gpu上.
    trt::Memory<float> m_image_embed_buffer;
    trt::Memory<float> m_high_res_feats0_buffer;
    trt::Memory<float> m_high_res_feats1_buffer;

    // stream
    cudaStream_t m_stream = nullptr;

    bool is_image_embedding_ready_ = false;

    const int target_size_ = 1024;

    bool initialize_trt(const std::string& model_path);

    template <typename T>
    T vectorProduct(const std::vector<T>& v) {
      return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
    }
  };
}

#endif //SEGMENT_ANYTHING2_SAM2_ENCODER_H
