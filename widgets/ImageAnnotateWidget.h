#ifndef AUTOLABELPROJECT_WIDGETS_IMAGEANNOTATEWIDGET_H
#define AUTOLABELPROJECT_WIDGETS_IMAGEANNOTATEWIDGET_H

#include <QImage>
#include <QList>
#include <QPoint>
#include <QRect>
#include <QVector>
#include <QWidget>

#include "app/AppTypes.h"

class QWheelEvent;
class QPainter;

class ImageAnnotateWidget : public QWidget {
    Q_OBJECT
public:
    explicit ImageAnnotateWidget(QWidget* parent = nullptr);

    bool loadImage(const QString& imagePath);
    void clearTempResult();
    void setTempResult(const TempInferenceResult& result);
    void setAnnotations(const QList<AnnotationObject>& annotations);
    void setPendingPromptRects(const QVector<QRectF>& rects);
    void clearPendingPromptRects();
    void setSelectedAnnotationIndex(int index);
    void setShowAnnotationLabels(bool show);
    QString viewportStatusText() const;

signals:
    void pointPromptRequested(const QPointF& imagePoint);
    void rectPromptRequested(const QRectF& imageRect);
    void annotationSelectionChanged(int annotationIndex);
    void viewportStatusChanged(const QString& statusText);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;

private:
    QRect imageDisplayRect() const;
    void emitViewportStatus();
    void drawEmptyState(QPainter* painter);
    void drawCanvasBackground(QPainter* painter);
    void drawHud(QPainter* painter, const QRect& display);
    bool widgetToImage(const QPoint& widgetPoint, QPointF* imagePoint) const;
    QPointF imageToWidget(const QPointF& imagePoint) const;
    QPolygonF imageToWidgetPolygon(const QPolygonF& polyImage) const;

    int findAnnotationAtImagePoint(const QPointF& imagePoint) const;
    QRectF normalizedImageRect(const QPointF& p1, const QPointF& p2) const;

    QImage m_image;
    QList<AnnotationObject> m_annotations;
    QVector<QRectF> m_pendingPromptRects;
    int m_selectedAnnotationIndex = -1;
    bool m_showAnnotationLabels = true;

    TempInferenceResult m_tempResult;

    bool m_mousePressed = false;
    bool m_draggingRect = false;
    QPoint m_pressWidgetPos;
    QPointF m_pressImagePos;
    QPoint m_currentWidgetPos;

    double m_zoomFactor = 1.0;
    QPointF m_viewOffsetWidget = QPointF(0.0, 0.0);
};

#endif // AUTOLABELPROJECT_WIDGETS_IMAGEANNOTATEWIDGET_H
