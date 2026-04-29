/**
  @Author: mpj
  @Date  : 2026/4/6 18:34
  @version V1.0
  @since C++17
**/
#include "automatic_mask_generator.h"

#include <utility>

Sam2::SamAutomaticMaskGenerator::SamAutomaticMaskGenerator(std::shared_ptr<Encoder> encoder,
                                                          std::shared_ptr<Decoder> decoder,
                                                          SamAutomaticMaskGeneratorParams params)
    : params_(std::move(params)), encoder_(std::move(encoder)), decoder_(std::move(decoder)) {
}

std::vector<Sam2::SamMask> Sam2::SamAutomaticMaskGenerator::forward(const cv::Mat& image) {
    std::vector<SamMask> all_masks;

    auto image_embedding = encoder_->forward(image);

    auto points = generate_grid_points(params_.points_per_side);

    // 处理主图像
    auto main_masks = process_points(image, points, image_embedding);
    all_masks.insert(all_masks.end(), main_masks.begin(), main_masks.end());

    // 多尺度裁剪处理
    if (params_.crop_n_layers > 0) {
        auto crop_boxes = generate_crop_boxes(image, std::max(image.rows, image.cols));

        for (int i = 0; i < crop_boxes.size(); ++i) {
            const auto& crop_box = crop_boxes[i];

            // 提取裁剪区域
            cv::Rect roi(crop_box[0], crop_box[1], crop_box[2], crop_box[3]);
            cv::Mat cropped_image = image(roi).clone();

            // 为裁剪区域生成点
            int points_per_side_crop = params_.points_per_side /
                (params_.crop_n_points_downscale_factor * (i + 1));
            points_per_side_crop = std::max(4, points_per_side_crop); // 最小4个点

            auto crop_points = generate_grid_points(points_per_side_crop, crop_box[0], crop_box[1]);

            // 处理裁剪区域
            auto crop_masks = process_points(cropped_image, crop_points, image_embedding, crop_box);

            // 应用裁剪NMS
            if (!crop_masks.empty()) {
                crop_masks = remove_duplicate_masks(crop_masks, params_.crop_nms_thresh);
            }

            all_masks.insert(all_masks.end(), crop_masks.begin(), crop_masks.end());
        }
    }

    // 移除重复的掩码
    if (!all_masks.empty()) {
        all_masks = remove_duplicate_masks(all_masks, params_.box_nms_thresh);
    }

    // 移除小区域
    if (params_.min_mask_region_area > 0) {
        all_masks = remove_small_regions(all_masks, params_.min_mask_region_area);
    }

    // 按面积排序
    std::sort(all_masks.begin(), all_masks.end(),
              [](const SamMask& a, const SamMask& b) {
                  return a.area > b.area;
              });

    return all_masks;
}

std::vector<cv::Point2f> Sam2::SamAutomaticMaskGenerator::generate_grid_points(int side_length, int crop_box_x,
                                                                              int crop_box_y) const {
    std::vector<cv::Point2f> points;

    // 生成均匀网格点
    for (int i = 0; i < side_length; ++i) {
        for (int j = 0; j < side_length; ++j) {
            float x = (i + 0.5f) * (target_size_ / side_length) + crop_box_x;
            float y = (j + 0.5f) * (target_size_ / side_length) + crop_box_y;
            points.emplace_back(x, y);
        }
    }

    return points;
}

std::vector<std::vector<int>> Sam2::SamAutomaticMaskGenerator::generate_crop_boxes(
    const cv::Mat& image, int orig_size) const {
    std::vector<std::vector<int>> crop_boxes;

    int image_height = image.rows;
    int image_width = image.cols;

    float crop_size = static_cast<float>(orig_size) * params_.crop_overlap_ratio;
    crop_size = std::min(static_cast<float>(std::min(image_height, image_width)), crop_size);

    // 生成重叠的裁剪框
    float stride = crop_size * (1.0f - params_.crop_overlap_ratio);

    for (float y = 0; y + crop_size <= image_height; y += stride) {
        for (float x = 0; x + crop_size <= image_width; x += stride) {
            crop_boxes.push_back({
                static_cast<int>(x),
                static_cast<int>(y),
                static_cast<int>(crop_size),
                static_cast<int>(crop_size)
            });
        }
    }

    return crop_boxes;
}

std::vector<Sam2::SamMask> Sam2::SamAutomaticMaskGenerator::process_points(const cv::Mat& image,
                                                                         const std::vector<cv::Point2f>& points,
                                                                         const ImageEmbedding& image_embedding,
                                                                         const std::vector<int>& crop_box) {
    std::vector<SamMask> masks;

    // 分批处理点
    int batch_size = static_cast<int>(params_.points_per_batch);
    for (size_t i = 0; i < points.size(); i += batch_size) {
        size_t end_idx = std::min(i + batch_size, points.size());
        PromptPoints batch_points;

        // 转换点格式
        for (size_t j = i; j < end_idx; ++j) {
            batch_points.push_back({points[j].x, points[j].y, 1}); // 前景点
        }

        try {
            // 运行解码器
            auto results = decoder_->forward_logits(image.size(), image_embedding, batch_points);
            SamResult result;
            for (auto item : results) {
                for (int index = 0; index < item.masks.size(); ++index) {
                    result.masks.emplace_back(item.masks[index]);
                    result.scores.emplace_back(item.scores[index]);
                }
            }
            // 处理每个掩码
            for (size_t j = 0; j < result.masks.size(); ++j) {
                float score = result.scores[j];

                // 应用IoU阈值
                if (score < params_.pred_iou_thresh) {
                    continue;
                }
                // 计算稳定性分数
                float stability_score = calculate_stability_score(
                    result.masks[j], 0.0f, params_.stability_score_offset);

                // 应用稳定性阈值
                if (stability_score < params_.stability_score_thresh) {
                    continue;
                }
                SamMask mask_info;

                // 创建二值掩码
                cv::Mat binary_mask;
                cv::threshold(result.masks[j], binary_mask, 0, 255, cv::THRESH_BINARY);
                binary_mask.convertTo(binary_mask, CV_8UC1);

                // 计算面积
                mask_info.area = cv::countNonZero(binary_mask);

                // 计算边界框
                std::vector<std::vector<cv::Point>> contours;
                cv::findContours(binary_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
                // 找到最大的轮廓（避免多个小区域）
                auto largest_contour = std::max_element(
                    contours.begin(), contours.end(),
                    [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                        return cv::contourArea(a) < cv::contourArea(b);
                    });
                mask_info.bbox = cv::boundingRect(*largest_contour);

                // 存储预测分数和稳定性分数
                mask_info.predicted_iou = score;
                mask_info.stability_score = stability_score;

                // 存储提示点坐标
                mask_info.point_coords[0] = static_cast<int>(batch_points[j % batch_points.size()].x);
                mask_info.point_coords[1] = static_cast<int>(batch_points[j % batch_points.size()].y);

                // 存储裁剪框信息
                if (!crop_box.empty()) {
                    mask_info.crop_box[0] = crop_box[0];
                    mask_info.crop_box[1] = crop_box[1];
                    mask_info.crop_box[2] = crop_box[2];
                    mask_info.crop_box[3] = crop_box[3];
                }

                // 存储分割结果
                cv::Mat mask_u8;
                cv::Mat threshold_mask = result.masks[j] > mask_threshold_;
                threshold_mask.convertTo(mask_u8, CV_8UC1);
                mask_info.segmentation["mask"] = mask_u8;

                masks.push_back(mask_info);
            }
        }
        catch (const std::exception& e) {
            std::cerr << "[Error] Error processing batch: " << e.what() << std::endl;
        }
    }

    return masks;
}

float Sam2::SamAutomaticMaskGenerator::calculate_stability_score(const cv::Mat& mask, float mask_threshold,
                                                                float threshold_offset) {
    // 计算掩码在不同阈值下的稳定性
    cv::Mat intersections, unions;

    // 使用两个不同的阈值
    cv::Mat mask1, mask2;
    cv::threshold(mask, mask1, mask_threshold - threshold_offset, 1.0f, cv::THRESH_BINARY);
    cv::threshold(mask, mask2, mask_threshold + threshold_offset, 1.0f, cv::THRESH_BINARY);

    mask1.convertTo(mask1, CV_8UC1);
    mask2.convertTo(mask2, CV_8UC1);

    // 计算交集和并集
    cv::Mat intersection, union_mat;
    cv::bitwise_and(mask1, mask2, intersection);
    cv::bitwise_or(mask1, mask2, union_mat);

    double intersection_area = cv::countNonZero(intersection);
    double union_area = cv::countNonZero(union_mat);

    if (union_area == 0) return 0.0f;
    return static_cast<float>(intersection_area / union_area);
}

std::vector<Sam2::SamMask> Sam2::SamAutomaticMaskGenerator::remove_duplicate_masks(std::vector<SamMask>& masks,
    float iou_threshold) {
    if (masks.empty()) return masks;

    // 提取边界框和分数
    std::vector<cv::Mat> boxes;
    std::vector<float> scores;

    for (const auto& mask : masks) {
        cv::Mat box(1, 4, CV_32FC1);
        box.at<float>(0) = static_cast<float>(mask.bbox.x);
        box.at<float>(1) = static_cast<float>(mask.bbox.y);
        box.at<float>(2) = static_cast<float>(mask.bbox.x + mask.bbox.width);
        box.at<float>(3) = static_cast<float>(mask.bbox.y + mask.bbox.height);
        boxes.push_back(box);
        scores.push_back(mask.predicted_iou * mask.stability_score); // 综合分数
    }

    // 应用NMS
    auto keep = batched_nms(boxes, scores, iou_threshold);

    // 保留被选中的掩码
    std::vector<SamMask> filtered_masks;
    for (size_t i = 0; i < masks.size(); ++i) {
        if (keep[i]) {
            filtered_masks.push_back(masks[i]);
        }
    }

    return filtered_masks;
}

std::vector<Sam2::SamMask> Sam2::SamAutomaticMaskGenerator::remove_small_regions(std::vector<SamMask>& masks,
                                                                               int min_area) {
    std::vector<SamMask> filtered_masks;

    for (const auto& mask : masks) {
        if (mask.area >= min_area) {
            filtered_masks.push_back(mask);
        }
    }

    return filtered_masks;
}

float Sam2::SamAutomaticMaskGenerator::calculate_mask_iou(const cv::Mat& mask1, const cv::Mat& mask2) {
    cv::Mat binary1, binary2;
    cv::threshold(mask1, binary1, 0.0f, 1.0f, cv::THRESH_BINARY);
    cv::threshold(mask2, binary2, 0.0f, 1.0f, cv::THRESH_BINARY);

    binary1.convertTo(binary1, CV_8UC1);
    binary2.convertTo(binary2, CV_8UC1);

    cv::Mat intersection, union_mat;
    cv::bitwise_and(binary1, binary2, intersection);
    cv::bitwise_or(binary1, binary2, union_mat);

    double intersection_area = cv::countNonZero(intersection);
    double union_area = cv::countNonZero(union_mat);

    return union_area > 0 ? static_cast<float>(intersection_area / union_area) : 0.0f;
}

std::vector<bool> Sam2::SamAutomaticMaskGenerator::batched_nms(const std::vector<cv::Mat>& boxes,
                                                              const std::vector<float>& scores, float iou_threshold) {
    return nms(boxes, scores, iou_threshold);
}

std::vector<bool> Sam2::SamAutomaticMaskGenerator::nms(const std::vector<cv::Mat>& boxes,
                                                      const std::vector<float>& scores, float iou_threshold) {
    if (boxes.empty()) return {};

    std::vector<bool> keep(boxes.size(), true);
    std::vector<size_t> indices(boxes.size());
    std::iota(indices.begin(), indices.end(), 0);

    // 按分数排序
    std::sort(indices.begin(), indices.end(),
              [&](size_t i, size_t j) { return scores[i] > scores[j]; });

    for (size_t i = 0; i < indices.size(); ++i) {
        if (!keep[indices[i]]) continue;

        for (size_t j = i + 1; j < indices.size(); ++j) {
            if (!keep[indices[j]]) continue;

            // 计算IoU
            const auto& box1 = boxes[indices[i]];
            const auto& box2 = boxes[indices[j]];

            float x1 = std::max(box1.at<float>(0), box2.at<float>(0));
            float y1 = std::max(box1.at<float>(1), box2.at<float>(1));
            float x2 = std::min(box1.at<float>(2), box2.at<float>(2));
            float y2 = std::min(box1.at<float>(3), box2.at<float>(3));

            float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);

            float area1 = (box1.at<float>(2) - box1.at<float>(0)) * (box1.at<float>(3) - box1.at<float>(1));
            float area2 = (box2.at<float>(2) - box2.at<float>(0)) * (box2.at<float>(3) - box2.at<float>(1));
            float union_area = area1 + area2 - intersection;

            float iou = intersection / union_area;

            if (iou > iou_threshold) {
                keep[indices[j]] = false;
            }
        }
    }

    return keep;
}
