/**
  @Author: mpj
  @Date  : 2026/4/6 18:15
  @version V1.0
  @since C++17
**/
#include "sam2.h"

Sam2::SAM::SAM(const std::string& encoder_model_path, const std::string& decoder_model_path, bool use_gpu,
int cuda_device_id) {
    encoder_ = std::make_shared<Encoder>(encoder_model_path, cuda_device_id);
    decoder_ = std::make_shared<Decoder>(decoder_model_path, cuda_device_id);
}

Sam2::Results Sam2::SAM::forward(const cv::Mat& bgr_image, const PromptPoints& points,
                                 const PromptBboxes& bboxes) const {
    ImageEmbedding image_embedding{};
    if (!encoder_->is_image_embedding_ready()) {
        image_embedding = encoder_->forward(bgr_image);
    } else {
        image_embedding = encoder_->get_image_embedding();
    }
    auto results = decoder_->forward(bgr_image.size(), image_embedding, points, bboxes);
    return results;
}

Sam2::Results Sam2::SAM::forward(const cv::Mat& bgr_image, const PromptBboxes& bboxes) const {
    ImageEmbedding image_embedding{};
    if (!encoder_->is_image_embedding_ready()) {
        image_embedding = encoder_->forward(bgr_image);
    } else {
        image_embedding = encoder_->get_image_embedding();
    }
    auto results = decoder_->forward(bgr_image.size(), image_embedding, bboxes);
    return results;
}

bool Sam2::SAM::reset() const {
    encoder_->reset_image_embedding();
    return true;
}
