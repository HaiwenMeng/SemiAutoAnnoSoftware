#include "inference/SamInferenceBridge.h"

#include <cmath>
#include <exception>

#include <QCoreApplication>
#include <QDebug>
#include <QDir>
#include <QFileInfo>
#include <QList>
#include <QStringList>

#include "trtsam3lib.h"

namespace {
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
