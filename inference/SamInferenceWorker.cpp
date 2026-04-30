#include "inference/SamInferenceWorker.h"

#include <exception>

#include <QDebug>

SamInferenceWorker::SamInferenceWorker(QObject* parent) : QObject(parent) {}

SamInferenceWorker::~SamInferenceWorker() = default;

void SamInferenceWorker::initialize() {
    QString error;
    const bool ok = m_bridge.initialize(&error);
    emit initializeFinished(ok, error);
}

void SamInferenceWorker::setCurrentImage(const QString& imagePath) {
    QString error;
    const bool ok = m_bridge.setCurrentImage(imagePath, &error);
    emit currentImageFinished(imagePath, ok, error);
}

void SamInferenceWorker::inferByPoint(const QPointF& imagePoint, const QString& labelName, const QString& imagePath) {
    SamInferResult result;
    try {
        result = m_bridge.inferByPoint(imagePoint, labelName);
    } catch (const std::exception& ex) {
        result.success = false;
        result.errorMessage = QStringLiteral("Infer by point exception: %1").arg(ex.what());
        qWarning().noquote() << "[InferByPoint]" << result.errorMessage;
    } catch (...) {
        result.success = false;
        result.errorMessage = QStringLiteral("Infer by point exception: unknown exception");
        qWarning().noquote() << "[InferByPoint]" << result.errorMessage;
    }
    emit pointInferenceFinished(result, labelName, imagePath);
}

void SamInferenceWorker::inferByRect(const QRectF& imageRect, const QString& labelName, const QString& imagePath) {
    SamInferResult result;
    try {
        result = m_bridge.inferByRect(imageRect, labelName);
    } catch (const std::exception& ex) {
        result.success = false;
        result.errorMessage = QStringLiteral("Infer by rect exception: %1").arg(ex.what());
        qWarning().noquote() << "[InferByRect]" << result.errorMessage;
    } catch (...) {
        result.success = false;
        result.errorMessage = QStringLiteral("Infer by rect exception: unknown exception");
        qWarning().noquote() << "[InferByRect]" << result.errorMessage;
    }
    emit rectInferenceFinished(result, labelName, imagePath);
}

void SamInferenceWorker::inferByRects(const QVector<QRectF>& imageRects, const QString& labelName,
                                      const QString& imagePath) {
    SamInferResult result;
    try {
        result = m_bridge.inferByRects(imageRects, labelName);
    } catch (const std::exception& ex) {
        result.success = false;
        result.errorMessage = QStringLiteral("Infer by rects exception: %1").arg(ex.what());
        qWarning().noquote() << "[InferByRects]" << result.errorMessage;
    } catch (...) {
        result.success = false;
        result.errorMessage = QStringLiteral("Infer by rects exception: unknown exception");
        qWarning().noquote() << "[InferByRects]" << result.errorMessage;
    }
    emit rectsInferenceFinished(result, labelName, imagePath);
}

void SamInferenceWorker::inferSmallTargetByRect(const QRectF& imageRect, const QString& labelName,
                                                const QString& imagePath) {
    SamInferResult result;
    try {
        result = m_bridge.inferSmallTargetByRect(imageRect, labelName);
    } catch (const std::exception& ex) {
        result.success = false;
        result.errorMessage = QStringLiteral("Small target infer exception: %1").arg(ex.what());
        qWarning().noquote() << "[InferSmallTarget]" << result.errorMessage;
    } catch (...) {
        result.success = false;
        result.errorMessage = QStringLiteral("Small target infer exception: unknown exception");
        qWarning().noquote() << "[InferSmallTarget]" << result.errorMessage;
    }
    emit rectInferenceFinished(result, labelName, imagePath);
}
