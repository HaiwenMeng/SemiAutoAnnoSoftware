#include "widgets/ImageAnnotateWidget.h"

#include <QMouseEvent>
#include <QWheelEvent>
#include <QPainter>
#include <QPainterPath>
#include <QFontMetrics>
#include <QtMath>

namespace {
QColor colorFromInt(int value) {
    return QColor((value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF);
}
}

ImageAnnotateWidget::ImageAnnotateWidget(QWidget* parent) : QWidget(parent) {
    setMouseTracking(true);
    setMinimumSize(480, 360);
}

bool ImageAnnotateWidget::loadImage(const QString& imagePath) {
    QImage image;
    if (!image.load(imagePath)) {
        return false;
    }

    m_image = image;
    m_zoomFactor = 1.0;
    m_viewOffsetWidget = QPointF(0.0, 0.0);
    m_tempResult = TempInferenceResult{};
    m_selectedAnnotationIndex = -1;
    update();
    return true;
}

void ImageAnnotateWidget::clearTempResult() {
    m_tempResult = TempInferenceResult{};
    update();
}

void ImageAnnotateWidget::setTempResult(const TempInferenceResult& result) {
    m_tempResult = result;
    update();
}

void ImageAnnotateWidget::setAnnotations(const QList<AnnotationObject>& annotations) {
    m_annotations = annotations;
    if (m_selectedAnnotationIndex >= m_annotations.size()) {
        m_selectedAnnotationIndex = -1;
    }
    update();
}

void ImageAnnotateWidget::setSelectedAnnotationIndex(int index) {
    if (index < -1 || index >= m_annotations.size()) {
        m_selectedAnnotationIndex = -1;
    } else {
        m_selectedAnnotationIndex = index;
    }
    update();
}

void ImageAnnotateWidget::paintEvent(QPaintEvent* event) {
    Q_UNUSED(event);

    QPainter painter(this);
    painter.fillRect(rect(), QColor(26, 26, 26));

    if (m_image.isNull()) {
        painter.setPen(QColor(180, 180, 180));
        painter.drawText(rect(), Qt::AlignCenter, QString::fromUtf8(u8"请先打开图像或文件夹"));
        return;
    }

    const QRect display = imageDisplayRect();
    painter.drawImage(display, m_image);

    painter.setRenderHint(QPainter::Antialiasing, true);

    for (int i = 0; i < m_annotations.size(); ++i) {
        const bool selected = (i == m_selectedAnnotationIndex);
        const AnnotationObject& ann = m_annotations.at(i);
        const QColor baseColor = colorFromInt(ann.colorValue);
        const QColor drawColor = selected ? QColor(255, 225, 0) : baseColor;
        const QPolygonF widgetPoly = imageToWidgetPolygon(ann.rectPolygonImage);

        painter.setPen(QPen(drawColor, selected ? 3.0 : 2.0));
        painter.setBrush(Qt::NoBrush);
        painter.drawPolygon(widgetPoly);

        const QRectF box = widgetPoly.boundingRect();
        const QString tag = QStringLiteral("%1 %2").arg(i + 1).arg(ann.label);
        const QFontMetrics fm(painter.font());
        const QRect textRect = fm.boundingRect(tag).adjusted(-4, -2, 4, 2);
        QPoint textPos(static_cast<int>(qRound(box.left())), static_cast<int>(qRound(box.top())) - 2);
        if (textPos.y() - textRect.height() < 0) {
            textPos.setY(static_cast<int>(qRound(box.top())) + textRect.height() + 2);
        }
        if (textPos.x() + textRect.width() > width()) {
            textPos.setX(qMax(0, width() - textRect.width()));
        }

        const QRect bgRect(textPos.x(), textPos.y() - textRect.height(), textRect.width(), textRect.height());
        painter.setPen(Qt::NoPen);
        painter.setBrush(QColor(0, 0, 0, 110));
        painter.drawRect(bgRect);

        painter.setPen(baseColor);
        painter.setBrush(Qt::NoBrush);
        painter.drawText(bgRect.adjusted(4, 0, -4, 0), Qt::AlignVCenter | Qt::AlignLeft, tag);
    }

    if (m_tempResult.valid) {
        if (!m_tempResult.contourImage.isEmpty()) {
            painter.setPen(QPen(QColor(255, 90, 90), 2.0));
            painter.setBrush(QColor(255, 90, 90, 45));
            painter.drawPolygon(imageToWidgetPolygon(m_tempResult.contourImage));
        }

        if (!m_tempResult.minAreaRectImage.isEmpty()) {
            painter.setPen(QPen(QColor(255, 215, 0), 2.0, Qt::DashLine));
            painter.setBrush(Qt::NoBrush);
            painter.drawPolygon(imageToWidgetPolygon(m_tempResult.minAreaRectImage));
        }
    }

    if (m_tempResult.hasClick) {
        const QPointF p = imageToWidget(m_tempResult.clickPointImage);
        painter.setPen(Qt::NoPen);
        painter.setBrush(QColor(0, 255, 0));
        painter.drawEllipse(p, 4.0, 4.0);
    }

    if (m_draggingRect && m_mousePressed) {
        QRect r(m_pressWidgetPos, m_currentWidgetPos);
        r = r.normalized();
        painter.setPen(QPen(QColor(80, 180, 255), 2.0, Qt::DashLine));
        painter.setBrush(QColor(80, 180, 255, 35));
        painter.drawRect(r);
    } else if (m_tempResult.hasRect && m_tempResult.promptRectImage.isValid()) {
        const QPointF p1 = imageToWidget(m_tempResult.promptRectImage.topLeft());
        const QPointF p2 = imageToWidget(m_tempResult.promptRectImage.bottomRight());
        QRectF wr(p1, p2);
        wr = wr.normalized();
        painter.setPen(QPen(QColor(80, 180, 255), 2.0, Qt::DashLine));
        painter.setBrush(Qt::NoBrush);
        painter.drawRect(wr);
    }
}

void ImageAnnotateWidget::mousePressEvent(QMouseEvent* event) {
    if (event->button() == Qt::RightButton && !m_image.isNull()) {
        m_zoomFactor = 1.0;
        m_viewOffsetWidget = QPointF(0.0, 0.0);
        update();
        event->accept();
        return;
    }

    if (event->button() != Qt::LeftButton || m_image.isNull()) {
        QWidget::mousePressEvent(event);
        return;
    }

    QPointF imagePoint;
    if (!widgetToImage(event->pos(), &imagePoint)) {
        QWidget::mousePressEvent(event);
        return;
    }

    m_mousePressed = true;
    m_draggingRect = false;
    m_pressWidgetPos = event->pos();
    m_currentWidgetPos = event->pos();
    m_pressImagePos = imagePoint;

    const int hitIndex = findAnnotationAtImagePoint(imagePoint);
    if (hitIndex != -1) {
        m_selectedAnnotationIndex = hitIndex;
        emit annotationSelectionChanged(hitIndex);
        update();
    }

    QWidget::mousePressEvent(event);
}

void ImageAnnotateWidget::wheelEvent(QWheelEvent* event) {
    if (m_image.isNull()) {
        QWidget::wheelEvent(event);
        return;
    }

    const QPoint delta = event->angleDelta();
    if (delta.y() == 0) {
        QWidget::wheelEvent(event);
        return;
    }

    QPointF anchorImage;
    QPoint anchorWidget = event->pos();
    if (!widgetToImage(anchorWidget, &anchorImage)) {
        anchorImage = QPointF((m_image.width() - 1) * 0.5, (m_image.height() - 1) * 0.5);
        anchorWidget = rect().center();
    }

    const double steps = static_cast<double>(delta.y()) / 120.0;
    const double scalePerStep = 1.15;
    m_zoomFactor *= qPow(scalePerStep, steps);
    m_zoomFactor = qBound(0.1, m_zoomFactor, 20.0);

    const QPointF anchorWidgetAfterZoom = imageToWidget(anchorImage);
    m_viewOffsetWidget += (QPointF(anchorWidget) - anchorWidgetAfterZoom);

    update();
    event->accept();
}

void ImageAnnotateWidget::mouseMoveEvent(QMouseEvent* event) {
    if (!m_mousePressed || m_image.isNull()) {
        QWidget::mouseMoveEvent(event);
        return;
    }

    m_currentWidgetPos = event->pos();
    const int manhattan = (m_currentWidgetPos - m_pressWidgetPos).manhattanLength();
    if (manhattan >= 6) {
        m_draggingRect = true;
        update();
    }

    QWidget::mouseMoveEvent(event);
}

void ImageAnnotateWidget::mouseReleaseEvent(QMouseEvent* event) {
    if (event->button() != Qt::LeftButton || !m_mousePressed || m_image.isNull()) {
        QWidget::mouseReleaseEvent(event);
        return;
    }

    QPointF releaseImage;
    const bool releaseValid = widgetToImage(event->pos(), &releaseImage);
    const bool emitRect = m_draggingRect && releaseValid;

    m_mousePressed = false;
    m_draggingRect = false;

    if (emitRect) {
        const QRectF imageRect = normalizedImageRect(m_pressImagePos, releaseImage);
        if (imageRect.width() >= 2.0 && imageRect.height() >= 2.0) {
            emit rectPromptRequested(imageRect);
        }
    } else {
        QPointF clickImage;
        if (widgetToImage(event->pos(), &clickImage)) {
            emit pointPromptRequested(clickImage);
        }
    }

    update();
    QWidget::mouseReleaseEvent(event);
}

QRect ImageAnnotateWidget::imageDisplayRect() const {
    if (m_image.isNull() || width() <= 0 || height() <= 0) {
        return {};
    }

    const QSize imageSize = m_image.size();
    const QSize areaSize = size();

    const double sx = static_cast<double>(areaSize.width()) / static_cast<double>(imageSize.width());
    const double sy = static_cast<double>(areaSize.height()) / static_cast<double>(imageSize.height());
    const double scale = qMin(sx, sy) * m_zoomFactor;

    const int drawW = qMax(1, static_cast<int>(qRound(imageSize.width() * scale)));
    const int drawH = qMax(1, static_cast<int>(qRound(imageSize.height() * scale)));
    const int x = static_cast<int>(qRound((areaSize.width() - drawW) / 2.0 + m_viewOffsetWidget.x()));
    const int y = static_cast<int>(qRound((areaSize.height() - drawH) / 2.0 + m_viewOffsetWidget.y()));
    return QRect(x, y, drawW, drawH);
}

bool ImageAnnotateWidget::widgetToImage(const QPoint& widgetPoint, QPointF* imagePoint) const {
    if (m_image.isNull() || imagePoint == nullptr) {
        return false;
    }

    const QRect display = imageDisplayRect();
    if (!display.contains(widgetPoint)) {
        return false;
    }

    const double nx = static_cast<double>(widgetPoint.x() - display.left()) / static_cast<double>(display.width());
    const double ny = static_cast<double>(widgetPoint.y() - display.top()) / static_cast<double>(display.height());

    const double x = nx * static_cast<double>(m_image.width() - 1);
    const double y = ny * static_cast<double>(m_image.height() - 1);
    *imagePoint = QPointF(x, y);
    return true;
}

QPointF ImageAnnotateWidget::imageToWidget(const QPointF& imagePoint) const {
    const QRect display = imageDisplayRect();
    if (m_image.isNull() || display.isEmpty()) {
        return {};
    }

    const double nx = imagePoint.x() / qMax(1, m_image.width() - 1);
    const double ny = imagePoint.y() / qMax(1, m_image.height() - 1);

    const double wx = display.left() + nx * static_cast<double>(display.width());
    const double wy = display.top() + ny * static_cast<double>(display.height());
    return QPointF(wx, wy);
}

QPolygonF ImageAnnotateWidget::imageToWidgetPolygon(const QPolygonF& polyImage) const {
    QPolygonF polyWidget;
    polyWidget.reserve(polyImage.size());
    for (const QPointF& p : polyImage) {
        polyWidget.push_back(imageToWidget(p));
    }
    return polyWidget;
}

int ImageAnnotateWidget::findAnnotationAtImagePoint(const QPointF& imagePoint) const {
    for (int i = m_annotations.size() - 1; i >= 0; --i) {
        if (m_annotations[i].rectPolygonImage.containsPoint(imagePoint, Qt::OddEvenFill)) {
            return i;
        }
    }
    return -1;
}

QRectF ImageAnnotateWidget::normalizedImageRect(const QPointF& p1, const QPointF& p2) const {
    QRectF r(p1, p2);
    r = r.normalized();

    const double maxX = qMax(0, m_image.width() - 1);
    const double maxY = qMax(0, m_image.height() - 1);

    r.setLeft(qBound(0.0, r.left(), maxX));
    r.setRight(qBound(0.0, r.right(), maxX));
    r.setTop(qBound(0.0, r.top(), maxY));
    r.setBottom(qBound(0.0, r.bottom(), maxY));
    return r;
}
