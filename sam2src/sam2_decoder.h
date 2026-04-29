/**
  @Author: mpj
  @Date  : 2026/4/6 17:35
  @version V1.0
  @since C++17
**/

#ifndef SEGMENT_ANYTHING2_SAM2_DECODER_H
#define SEGMENT_ANYTHING2_SAM2_DECODER_H

#include <numeric>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include "infer.h"
#include "sam2_encoder.h"

namespace Sam2 {
    class Decoder {
    public:
        explicit Decoder(const std::string& model_path, int cuda_device_id = 0);

        ~Decoder();

        /**
         * 只对point进行prompt，返回多个mask结果
         * 1. 可以只使用point进行prompt
         * 2. 如果使用point+box进行prompt，推荐只使用一个box，多个box可能会导致分割能力下降
         */
        Results forward(const cv::Size& original_image_size, const ImageEmbedding& image_embedding,
                        const PromptPoints& prompt_points, const PromptBboxes& prompt_bboxes = {});

        /**
         * 利用batch推理的方式对多个box进行prompt，返回多个mask结果
         * 1. 不可以传入point进行prompt，必须使用box进行prompt
         * 2. 可以传入多个box进行prompt
         * 3. 返回box * 4个mask，更具score排序，score越高mask质量越好，score为mask的置信度，范围[0, 1]
         */
        Results forward(const cv::Size& original_image_size, const ImageEmbedding& image_embedding,
                        const PromptBboxes& prompt_bboxes);

        /**
         * batch推理point
         */
        Results forward_logits(const cv::Size& original_image_size, const ImageEmbedding& image_embedding,
                               const PromptPoints& prompt_points);

    private:
        std::vector<std::string> input_names;
        std::vector<std::string> output_names;
        std::vector<int> m_image_embed_dims;
        std::vector<int> m_high_res_feats0_dims;
        std::vector<int> m_high_res_feats1_dims;
        std::vector<int> m_point_coords_dims;
        std::vector<int> m_point_labels_dims;
        std::vector<int> m_mask_input_dims;
        std::vector<int> m_has_mask_input_dims;
        std::vector<int> m_iou_predictions_dims;
        std::vector<int> m_low_res_masks_dims;

        std::shared_ptr<trt::Infer> m_trt;
        // 预处理缓存，用来存储预处理之后的图片，放在gpu上
        trt::Memory<float> m_point_coords_buffer;
        trt::Memory<float> m_point_labels_buffer;
        trt::Memory<float> m_mask_input_buffer;
        trt::Memory<float> m_has_mask_input_buffer;
        // 输出缓存，用来存储输出结果，放在gpu上.
        trt::Memory<float> m_iou_predictions_buffer;
        trt::Memory<float> m_low_res_masks_buffer;

        // stream
        cudaStream_t m_stream = nullptr;

        const int target_size_ = 1024;
        const float mask_threshold_ = 0.0f;

        bool initialize_trt(const std::string& model_path);

        void adjust_memory(int batch_size, int num_points);

        template <typename T>
        T vectorProduct(const std::vector<T>& v) {
            return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
        }

        cv::Mat mask_postprocessing(cv::Mat& mask, const cv::Size& original_image_size) const;
    };
}

#endif //SEGMENT_ANYTHING2_SAM2_DECODER_H
