#include "inference/MaskPostProcessor.h"

#include <algorithm>

#include <opencv2/imgproc.hpp>

bool MaskPostProcessor::pickBestMask(const Sam2::Results& results, cv::Mat* outMask, QString* errorMessage) {
    if (outMask == nullptr) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("Internal error: outMask is null");
        }
        return false;
    }

    float bestScore = -1.0f;
    cv::Mat bestMask;

    for (const auto& batch : results) {
        const size_t itemCount = std::min(batch.masks.size(), batch.scores.size());
        for (size_t i = 0; i < itemCount; ++i) {
            const cv::Mat& candidate = batch.masks[i];
            if (candidate.empty()) {
                continue;
            }

            const float score = batch.scores[i];
            if (score > bestScore) {
                bestScore = score;
                bestMask = candidate;
            }
        }
    }

    if (bestMask.empty()) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("SAM2 returned no valid mask");
        }
        return false;
    }

    cv::Mat binary;
    if (bestMask.type() == CV_8UC1) {
        cv::threshold(bestMask, binary, 127, 255, cv::THRESH_BINARY);
    } else {
        cv::Mat temp;
        bestMask.convertTo(temp, CV_8UC1, 255.0);
        cv::threshold(temp, binary, 127, 255, cv::THRESH_BINARY);
    }

    *outMask = binary;
    return true;
}

bool MaskPostProcessor::extractLargestContour(const cv::Mat& binaryMask, QPolygonF* contourImage, QString* errorMessage) {
    if (contourImage == nullptr) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("Internal error: contourImage is null");
        }
        return false;
    }
    if (binaryMask.empty()) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("Mask is empty");
        }
        return false;
    }

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binaryMask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("No contour extracted from mask");
        }
        return false;
    }

    const auto maxIt = std::max_element(contours.begin(), contours.end(),
                                        [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                                            return cv::contourArea(a) < cv::contourArea(b);
                                        });

    if (maxIt == contours.end() || maxIt->empty()) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("No valid contour found in mask");
        }
        return false;
    }

    *contourImage = toQPolygonF(*maxIt);
    return !contourImage->isEmpty();
}

QPolygonF MaskPostProcessor::computeMinBoundingRect(const QPolygonF& contourImage) {
    if (contourImage.isEmpty()) {
        return {};
    }

    std::vector<cv::Point2f> points;
    points.reserve(contourImage.size());
    for (const QPointF& p : contourImage) {
        points.emplace_back(static_cast<float>(p.x()), static_cast<float>(p.y()));
    }

    const cv::Rect2f rect = cv::boundingRect(points);

    QPolygonF polygon;
    polygon.reserve(4);
    polygon.push_back(QPointF(rect.x, rect.y));
    polygon.push_back(QPointF(rect.x + rect.width, rect.y));
    polygon.push_back(QPointF(rect.x + rect.width, rect.y + rect.height));
    polygon.push_back(QPointF(rect.x, rect.y + rect.height));
    return polygon;
}

QPolygonF MaskPostProcessor::toQPolygonF(const std::vector<cv::Point>& contour) {
    QPolygonF polygon;
    polygon.reserve(static_cast<int>(contour.size()));
    for (const cv::Point& p : contour) {
        polygon.push_back(QPointF(p.x, p.y));
    }
    return polygon;
}