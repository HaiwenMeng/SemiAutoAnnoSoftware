#include "inference/SamInferenceBridge.h"

#include <cmath>
#include <exception>

#include <QCoreApplication>
#include <QDebug>
#include <QDir>
#include <QFileInfo>
#include <QList>
#include <QRect>
#include <QStringList>

#include "trtsam3lib.h"

namespace {
constexpr double kSmallTargetCropFactor = 20.0;
constexpr int kInferenceSizeMultiple = 4;

QString sam3ModelDir() {
    return QDir(QCoreApplication::applicationDirPath()).filePath(QStringLiteral("sam3"));
}

bool hasAnyCandidate(const QDir& dir, const QStringList& candidates) {
    for (const QString& name : candidates) {
        if (QFileInfo::exists(dir.filePath(name))) {
            return true;
        }
    }
    return false;
}

bool validateSam3ModelDir(const QString& modelDir, QString* errorMessage) {
    const QDir dir(modelDir);
    if (!dir.exists()) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("SAM3 model directory does not exist: %1").arg(modelDir);
        }
        return false;
    }

    const QList<QStringList> required = {
        {QStringLiteral("vision-encoder.fp16.trt"), QStringLiteral("vision-encoder.trt"), QStringLiteral("vision-encoder.engine")},
        {QStringLiteral("text-encoder.fp16.trt"), QStringLiteral("text-encoder.trt"), QStringLiteral("text-encoder.engine")},
        {QStringLiteral("geometry-encoder.fp16.trt"), QStringLiteral("geometry-encoder.trt"), QStringLiteral("geometry-encoder.engine")},
        {QStringLiteral("decoder.fp16.trt"), QStringLiteral("decoder.trt"), QStringLiteral("decoder.engine")}
    };

    for (const QStringList& candidates : required) {
        if (!hasAnyCandidate(dir, candidates)) {
            if (errorMessage) {
                *errorMessage = QStringLiteral("SAM3 model file missing in %1. Expected one of: %2")
                                    .arg(modelDir, candidates.join(QStringLiteral(", ")));
            }
            return false;
        }
    }

    return true;
}

int ceilToMultiple(double value, int multiple) {
    return qMax(multiple, static_cast<int>(std::ceil(value / multiple)) * multiple);
}

int floorToMultiple(int value, int multiple) {
    return (value / multiple) * multiple;
}

void translateRoi(QVector<double>* roiData, double dx, double dy) {
    if (roiData == nullptr) {
        return;
    }
    for (int i = 0; i + 1 < roiData->size(); i += 2) {
        (*roiData)[i] += dx;
        (*roiData)[i + 1] += dy;
    }
}

void translateContour(QVector<QPointF>* contour, double dx, double dy) {
    if (contour == nullptr) {
        return;
    }
    for (QPointF& point : *contour) {
        point += QPointF(dx, dy);
    }
}

void translateResult(SamInferResult* result, double dx, double dy) {
    if (result == nullptr) {
        return;
    }
    for (SamObjectResult& object : result->objects) {
        translateRoi(&object.roiData, dx, dy);
        translateRoi(&object.minRectRoiData, dx, dy);
        translateContour(&object.maskContour, dx, dy);
    }
}

bool makeSmallTargetCrop(const QRectF& imageRect, const QSize& imageSize, QRect* cropRect,
                         QRectF* localPromptRect, QString* errorMessage) {
    if (cropRect == nullptr || localPromptRect == nullptr) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("Internal error: null small target crop output.");
        }
        return false;
    }
    if (imageSize.width() < kInferenceSizeMultiple || imageSize.height() < kInferenceSizeMultiple) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("Image is too small for small target inference.");
        }
        return false;
    }

    const QRectF imageBounds(0.0, 0.0, imageSize.width(), imageSize.height());
    const QRectF prompt = imageRect.normalized().intersected(imageBounds);
    if (!prompt.isValid() || prompt.width() < 2.0 || prompt.height() < 2.0) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("Small target prompt rect is invalid or too small.");
        }
        return false;
    }

    const int maxCropWidth = floorToMultiple(imageSize.width(), kInferenceSizeMultiple);
    const int maxCropHeight = floorToMultiple(imageSize.height(), kInferenceSizeMultiple);
    if (maxCropWidth < kInferenceSizeMultiple || maxCropHeight < kInferenceSizeMultiple) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("Image dimensions cannot produce a 4-aligned crop.");
        }
        return false;
    }

    const int cropWidth = qMin(ceilToMultiple(prompt.width() * kSmallTargetCropFactor, kInferenceSizeMultiple),
                              maxCropWidth);
    const int cropHeight = qMin(ceilToMultiple(prompt.height() * kSmallTargetCropFactor, kInferenceSizeMultiple),
                               maxCropHeight);
    const QPointF center = prompt.center();
    const int left = qBound(0, static_cast<int>(std::round(center.x() - cropWidth * 0.5)),
                            imageSize.width() - cropWidth);
    const int top = qBound(0, static_cast<int>(std::round(center.y() - cropHeight * 0.5)),
                           imageSize.height() - cropHeight);

    *cropRect = QRect(left, top, cropWidth, cropHeight);
    *localPromptRect = prompt.translated(-left, -top);
    return true;
}
}

SamInferenceBridge::SamInferenceBridge() : m_sam3(new TrtSam3()) {}

SamInferenceBridge::~SamInferenceBridge() = default;

bool SamInferenceBridge::initialize(QString* errorMessage) {
    if (m_initialized) {
        return true;
    }

    const QString modelDir = sam3ModelDir();
    QString validationError;
    if (!validateSam3ModelDir(modelDir, &validationError)) {
        if (errorMessage) {
            *errorMessage = validationError;
        }
        qWarning().noquote() << "[SAM3 Init]" << validationError;
        return false;
    }

    try {
        m_initialized = m_sam3 && m_sam3->initialize(modelDir, 0);
    } catch (const std::exception& ex) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("Failed to initialize SAM3: %1").arg(ex.what());
        }
        m_initialized = false;
        qWarning() << "[SAM3 Init]" << ex.what();
        return false;
    } catch (...) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("Failed to initialize SAM3: unknown exception");
        }
        m_initialized = false;
        qWarning() << "[SAM3 Init] unknown exception";
        return false;
    }

    if (!m_initialized) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("Failed to initialize SAM3 with model directory: %1").arg(modelDir);
        }
        qWarning().noquote() << "[SAM3 Init] failed with model dir" << modelDir;
        return false;
    }

    if (!m_currentImage.isNull()) {
        m_sam3->setCurrentImage(m_currentImage);
    }

    return true;
}

bool SamInferenceBridge::isInitialized() const {
    return m_initialized;
}

bool SamInferenceBridge::setCurrentImage(const QString& imagePath, QString* errorMessage) {
    if (!m_initialized) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("SAM3 is not initialized");
        }
        return false;
    }

    QImage image;
    if (!image.load(imagePath)) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("Failed to read image: %1").arg(imagePath);
        }
        qWarning().noquote() << "[SAM3 SetCurrentImage] failed to read" << imagePath;
        return false;
    }

    m_currentImage = image;
    m_currentImagePath = imagePath;

    try {
        m_sam3->setCurrentImage(m_currentImage);
    } catch (const std::exception& ex) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("Failed to set SAM3 image: %1").arg(ex.what());
        }
        qWarning() << "[SAM3 SetCurrentImage]" << ex.what();
        return false;
    } catch (...) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("Failed to set SAM3 image: unknown exception");
        }
        qWarning() << "[SAM3 SetCurrentImage] unknown exception";
        return false;
    }

    return true;
}

SamInferResult SamInferenceBridge::inferByPoint(const QPointF& imagePoint, const QString& labelName) {
    SamInferResult result;
    if (!validateReady(&result)) {
        return result;
    }

    if (!std::isfinite(imagePoint.x()) || !std::isfinite(imagePoint.y())) {
        result.errorMessage = QStringLiteral("Invalid point prompt");
        qWarning().noquote() << "[SAM3 InferByPoint]" << result.errorMessage;
        return result;
    }

    result = m_sam3->inferByPoint(imagePoint, labelName);
    if (!result.success) {
        qWarning().noquote() << "[SAM3 InferByPoint]" << result.errorMessage;
    }
    return result;
}

SamInferResult SamInferenceBridge::inferByRect(const QRectF& imageRect, const QString& labelName) {
    SamInferResult result;
    if (!validateReady(&result)) {
        return result;
    }

    const QRectF normalized = imageRect.normalized();
    if (!normalized.isValid() || normalized.width() < 2.0 || normalized.height() < 2.0) {
        result.errorMessage = QStringLiteral("Rect prompt is too small");
        qWarning().noquote() << "[SAM3 InferByRect]" << result.errorMessage;
        return result;
    }

    result = m_sam3->inferByRect(normalized, labelName);
    if (!result.success) {
        qWarning().noquote() << "[SAM3 InferByRect]" << result.errorMessage;
    }
    return result;
}

SamInferResult SamInferenceBridge::inferByRects(const QVector<QRectF>& imageRects, const QString& labelName) {
    SamInferResult result;
    if (!validateReady(&result)) {
        return result;
    }
    if (imageRects.isEmpty()) {
        result.errorMessage = QStringLiteral("No rect prompts for batch inference");
        qWarning().noquote() << "[SAM3 InferByRects]" << result.errorMessage;
        return result;
    }

    QVector<QRectF> normalizedRects;
    normalizedRects.reserve(imageRects.size());
    for (const QRectF& rect : imageRects) {
        const QRectF normalized = rect.normalized();
        if (!normalized.isValid() || normalized.width() < 2.0 || normalized.height() < 2.0) {
            result.errorMessage = QStringLiteral("Batch rect prompt is too small");
            qWarning().noquote() << "[SAM3 InferByRects]" << result.errorMessage;
            return result;
        }
        normalizedRects.push_back(normalized);
    }

    result = m_sam3->inferByRects(normalizedRects, labelName);
    if (!result.success) {
        qWarning().noquote() << "[SAM3 InferByRects]" << result.errorMessage;
    }
    return result;
}

SamInferResult SamInferenceBridge::inferSmallTargetByRect(const QRectF& imageRect, const QString& labelName) {
    SamInferResult result;
    if (!validateReady(&result)) {
        return result;
    }

    QRect cropRect;
    QRectF localPromptRect;
    QString cropError;
    if (!makeSmallTargetCrop(imageRect, m_currentImage.size(), &cropRect, &localPromptRect, &cropError)) {
        result.errorMessage = cropError;
        qWarning().noquote() << "[SAM3 SmallTarget]" << result.errorMessage;
        return result;
    }

    const QImage cropImage = m_currentImage.copy(cropRect);
    if (cropImage.isNull()) {
        result.errorMessage = QStringLiteral("Failed to crop image for small target inference.");
        qWarning().noquote() << "[SAM3 SmallTarget]" << result.errorMessage;
        return result;
    }

    try {
        m_sam3->setCurrentImage(cropImage);
        result = m_sam3->inferByRect(localPromptRect, labelName);
    } catch (const std::exception& ex) {
        result.success = false;
        result.errorMessage = QStringLiteral("Small target inference failed: %1").arg(ex.what());
        qWarning().noquote() << "[SAM3 SmallTarget]" << result.errorMessage;
    } catch (...) {
        result.success = false;
        result.errorMessage = QStringLiteral("Small target inference failed: unknown exception");
        qWarning().noquote() << "[SAM3 SmallTarget]" << result.errorMessage;
    }

    QString restoreError;
    if (!restoreCurrentImage(&restoreError)) {
        result.success = false;
        result.errorMessage = result.errorMessage.isEmpty()
            ? restoreError
            : QStringLiteral("%1; %2").arg(result.errorMessage, restoreError);
        qWarning().noquote() << "[SAM3 SmallTarget]" << restoreError;
        return result;
    }

    if (!result.success) {
        if (result.errorMessage.isEmpty()) {
            result.errorMessage = QStringLiteral("Small target inference failed.");
        }
        qWarning().noquote() << "[SAM3 SmallTarget]" << result.errorMessage;
        return result;
    }

    translateResult(&result, cropRect.left(), cropRect.top());
    return result;
}

bool SamInferenceBridge::validateReady(SamInferResult* result) const {
    if (result == nullptr) {
        return false;
    }
    if (!m_initialized || !m_sam3 || !m_sam3->isInitialized()) {
        result->errorMessage = QStringLiteral("SAM3 is not initialized");
        qWarning().noquote() << "[SAM3 Infer]" << result->errorMessage;
        return false;
    }
    if (m_currentImage.isNull()) {
        result->errorMessage = QStringLiteral("Current image is not set");
        qWarning().noquote() << "[SAM3 Infer]" << result->errorMessage;
        return false;
    }
    return true;
}

bool SamInferenceBridge::restoreCurrentImage(QString* errorMessage) {
    if (!m_sam3 || m_currentImage.isNull()) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("Cannot restore SAM3 current image after small target inference.");
        }
        return false;
    }

    try {
        m_sam3->setCurrentImage(m_currentImage);
    } catch (const std::exception& ex) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("Failed to restore SAM3 current image: %1").arg(ex.what());
        }
        return false;
    } catch (...) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("Failed to restore SAM3 current image: unknown exception");
        }
        return false;
    }
    return true;
}
