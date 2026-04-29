#ifndef AUTOLABELPROJECT_INFERENCE_MASKPOSTPROCESSOR_H
#define AUTOLABELPROJECT_INFERENCE_MASKPOSTPROCESSOR_H

#include <QPolygonF>
#include <QString>

#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

#include "sam2.h"

class MaskPostProcessor {
public:
    static bool pickBestMask(const Sam2::Results& results, cv::Mat* outMask, QString* errorMessage = nullptr);
    static bool extractLargestContour(const cv::Mat& binaryMask, QPolygonF* contourImage, QString* errorMessage = nullptr);

    // Axis-aligned minimal outer rectangle (not rotated).
    static QPolygonF computeMinBoundingRect(const QPolygonF& contourImage);

private:
    static QPolygonF toQPolygonF(const std::vector<cv::Point>& contour);
};

#endif // AUTOLABELPROJECT_INFERENCE_MASKPOSTPROCESSOR_H