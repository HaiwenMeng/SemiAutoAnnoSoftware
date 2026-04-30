#ifndef AUTOLABELPROJECT_INFERENCE_SAMINFERENCEWORKER_H
#define AUTOLABELPROJECT_INFERENCE_SAMINFERENCEWORKER_H

#include <QObject>
#include <QPointF>
#include <QRectF>
#include <QString>
#include <QVector>

#include "inference/SamInferenceBridge.h"
#include "inference/SamTypes.h"

class SamInferenceWorker : public QObject {
    Q_OBJECT
public:
    explicit SamInferenceWorker(QObject* parent = nullptr);
    ~SamInferenceWorker() override;

public slots:
    void initialize();
    void setCurrentImage(const QString& imagePath);
    void inferByPoint(const QPointF& imagePoint, const QString& labelName, const QString& imagePath);
    void inferByRect(const QRectF& imageRect, const QString& labelName, const QString& imagePath);
    void inferByRects(const QVector<QRectF>& imageRects, const QString& labelName, const QString& imagePath);
    void inferSmallTargetByRect(const QRectF& imageRect, const QString& labelName, const QString& imagePath);

signals:
    void initializeFinished(bool success, const QString& errorMessage);
    void currentImageFinished(const QString& imagePath, bool success, const QString& errorMessage);
    void pointInferenceFinished(const SamInferResult& result, const QString& labelName, const QString& imagePath);
    void rectInferenceFinished(const SamInferResult& result, const QString& labelName, const QString& imagePath);
    void rectsInferenceFinished(const SamInferResult& result, const QString& labelName, const QString& imagePath);

private:
    SamInferenceBridge m_bridge;
};

Q_DECLARE_METATYPE(SamInferResult)
Q_DECLARE_METATYPE(QVector<QRectF>)

#endif // AUTOLABELPROJECT_INFERENCE_SAMINFERENCEWORKER_H
