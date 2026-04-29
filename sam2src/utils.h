/**
  @Author: mpj
  @Date  : 2026/4/6 17:11
  @version V1.0
  @since C++17
**/

#ifndef SEGMENT_ANYTHING2_UTILS_H
#define SEGMENT_ANYTHING2_UTILS_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include <string>
#include <string>

namespace Sam2 {
  struct Result {
    std::vector<cv::Mat> masks; // 0-255 uint8二值化矩阵
    std::vector<cv::Mat> low_res_masks;
    std::vector<float> scores;
  };

  using Results = std::vector<Result>;

  struct PromptPoint {
    float x, y;
    int label; // 1 for positive, 0 for negative
  };

  using PromptPoints = std::vector<PromptPoint>;

  struct PromptBbox {
    float x1, y1, x2, y2;
  };

  using PromptBboxes = std::vector<PromptBbox>;

  class ImagePreprocessor {
  public:
    /**
     * 将std::string转换为std::wstring
     */
    static std::wstring stringToWString(const std::string &str);

    /**
     * 图像直接缩放为target_size，转为fp32数据
     */
    static cv::Mat resize_fp32(const cv::Mat &image, int target_size = 1024);

    /**
     * 将mat格式的hwc排布转为chw，转为fp32数据
     */
    static cv::Mat hwc2chw_fp32(const cv::Mat &image);

    /**
     * 将坐标从原始图像尺寸转换到预处理后图像尺寸
     */
    static cv::Point2f apply_coords(const cv::Point2f &point, const cv::Size &original_size, int target_size);
  };
} // namespace Sam2

#endif //SEGMENT_ANYTHING2_UTILS_H
