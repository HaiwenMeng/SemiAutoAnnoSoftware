/**
  @Author: mpj
  @Date  : 2026/4/6 18:15
  @version V1.0
  @since C++17
**/

#ifndef SEGMENT_ANYTHING2_SAM2_H
#define SEGMENT_ANYTHING2_SAM2_H

#include "sam2_encoder.h"
#include "sam2_decoder.h"

namespace Sam2 {
  class SAM {
  public:
    explicit SAM(const std::string& encoder_model_path, const std::string& decoder_model_path,
                 bool use_gpu = true, int cuda_device_id = 0);

    ~SAM() = default;

    /**
     * 只对point进行prompt，返回多个mask结果
     * 1. 可以只使用point进行prompt，最少需要一个正点，负点可选
     * 2. 如果使用point+box进行prompt，推荐只使用一个box，多个box可能会导致分割能力下降
     * 3. 返回4个mask，更具score排序，score越高mask质量越好，score为mask的置信度，范围[0, 1]
     */
    Results forward(const cv::Mat& bgr_image, const PromptPoints& points, const PromptBboxes& bboxes = {}) const;

    /**
     * 利用batch推理的方式对多个box进行prompt，返回多个mask结果
     * 1. 不可以传入point进行prompt，必须使用box进行prompt
     * 2. 可以传入多个box进行prompt
     * 3. 返回box * 4个mask，更具score排序，score越高mask质量越好，score为mask的置信度，范围[0, 1]
     */
    Results forward(const cv::Mat& bgr_image, const PromptBboxes& bboxes) const;

    bool reset() const;

  private:
    std::shared_ptr<Decoder> decoder_;
    std::shared_ptr<Encoder> encoder_;
  };
}

#endif //SEGMENT_ANYTHING2_SAM2_H
