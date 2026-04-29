/**
  @Author: mpj
  @Date  : 2026/4/6 17:11
  @version V1.0
  @since C++17
**/
#include "utils.h"

#if WIN32
#include <Windows.h>
#endif

std::wstring Sam2::ImagePreprocessor::stringToWString(const std::string &str) {
  int size_needed = MultiByteToWideChar(CP_UTF8, 0, &str[0], static_cast<int>(str.size()), NULL, 0);
  std::wstring wstrTo(size_needed, 0);
  MultiByteToWideChar(CP_UTF8, 0, &str[0], static_cast<int>(str.size()), &wstrTo[0], size_needed);
  return wstrTo;
}

cv::Mat Sam2::ImagePreprocessor::resize_fp32(const cv::Mat &image, int target_size) {
  cv::Mat resized;
  image.convertTo(resized, CV_32FC3);
  cv::resize(resized, resized, cv::Size(target_size, target_size));
  return resized;
}

cv::Mat Sam2::ImagePreprocessor::hwc2chw_fp32(const cv::Mat &image) {
  cv::Mat chw;
  cv::dnn::blobFromImage(image, chw);
  return chw;
}

cv::Point2f Sam2::ImagePreprocessor::apply_coords(const cv::Point2f &point, const cv::Size &original_size,
                                                  int target_size) {
  float new_h = point.y / static_cast<float>(original_size.height) * static_cast<float>(target_size);
  float new_w = point.x / static_cast<float>(original_size.width) * static_cast<float>(target_size);
  cv::Point2f new_point(new_w, new_h);
  return new_point;
}
