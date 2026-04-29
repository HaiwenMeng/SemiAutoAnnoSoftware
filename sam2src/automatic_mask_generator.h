/**
  @Author: mpj
  @Date  : 2026/4/6 18:34
  @version V1.0
  @since C++17
**/

#ifndef SEGMENT_ANYTHING2_AUTOMATIC_MASK_GENERATOR_H
#define SEGMENT_ANYTHING2_AUTOMATIC_MASK_GENERATOR_H

#include <vector>
#include <memory>
#include <map>
#include "sam2_encoder.h"
#include "sam2_decoder.h"

namespace Sam2 {
    struct SamMask {
        std::map<std::string, cv::Mat> segmentation; // 包含mask和其他可能的分割结果
        float area;
        cv::Rect bbox;
        float predicted_iou;
        int point_coords[2]; // 用于生成此掩码的提示点坐标
        float stability_score;
        int crop_box[4]; // [x, y, width, height]
    };

    struct SamAutomaticMaskGeneratorParams {
        // 点生成参数
        int points_per_side = 32;
        float points_per_batch = 64;

        // 掩码过滤参数
        float pred_iou_thresh = 0.88f;
        float stability_score_thresh = 0.95f;
        float stability_score_offset = 1.0f;

        // 多尺度裁剪参数
        int crop_n_layers = 0;
        float crop_nms_thresh = 0.7f;
        float crop_overlap_ratio = 512.0f / 1500.0f;
        int crop_n_points_downscale_factor = 1;
        std::string crop_nms_thresh_name = "crop_nms_thresh";

        // 后处理参数
        float box_nms_thresh = 0.7f;
        int min_mask_region_area = 0;

        // 输出格式
        bool output_mode = false; // 是否以特定格式输出
    };

    struct SamResult {
        std::vector<cv::Mat> masks;
        std::vector<float> scores;
    };

    class SamAutomaticMaskGenerator {
    public:
        SamAutomaticMaskGenerator(
            std::shared_ptr<Encoder> encoder,
            std::shared_ptr<Decoder> decoder,
            SamAutomaticMaskGeneratorParams params = SamAutomaticMaskGeneratorParams()
        );

        std::vector<SamMask> forward(const cv::Mat& image);

        // 参数设置
        void set_points_per_side(int points) { params_.points_per_side = points; }
        void set_pred_iou_thresh(float thresh) { params_.pred_iou_thresh = thresh; }
        void set_stability_score_thresh(float thresh) { params_.stability_score_thresh = thresh; }
        void set_crop_n_layers(int layers) { params_.crop_n_layers = layers; }
        void set_min_mask_region_area(int area) { params_.min_mask_region_area = area; }
        void set_box_nms_thresh(float thresh) { params_.box_nms_thresh = thresh; }

    private:
        SamAutomaticMaskGeneratorParams params_;
        std::shared_ptr<Encoder> encoder_;
        std::shared_ptr<Decoder> decoder_;
        const float target_size_ = 1024.f;
        const float mask_threshold_ = 0.0f;

        // 核心方法
        std::vector<cv::Point2f> generate_grid_points(int side_length, int crop_box_x = 0, int crop_box_y = 0) const;
        std::vector<std::vector<int>> generate_crop_boxes(const cv::Mat& image, int orig_size) const;
        std::vector<SamMask> process_points(
            const cv::Mat& image,
            const std::vector<cv::Point2f>& points,
            const ImageEmbedding& image_embedding,
            const std::vector<int>& crop_box = {0, 0, 0, 0}
        );

        // 掩码处理
        float calculate_stability_score(const cv::Mat& mask, float mask_threshold, float threshold_offset);
        std::vector<SamMask> remove_duplicate_masks(std::vector<SamMask>& masks, float iou_threshold);
        std::vector<SamMask> remove_small_regions(std::vector<SamMask>& masks, int min_area);

        // 工具方法
        float calculate_mask_iou(const cv::Mat& mask1, const cv::Mat& mask2);

        // NMS 方法
        std::vector<bool> batched_nms(
            const std::vector<cv::Mat>& boxes,
            const std::vector<float>& scores,
            float iou_threshold
        );
        std::vector<bool> nms(
            const std::vector<cv::Mat>& boxes,
            const std::vector<float>& scores,
            float iou_threshold
        );
    };
}

#endif //SEGMENT_ANYTHING2_AUTOMATIC_MASK_GENERATOR_H
