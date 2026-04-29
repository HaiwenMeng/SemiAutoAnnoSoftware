#include "widgets/ImageAnnotateWidget.h"

#include <QFontMetrics>
#include <QLinearGradient>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QPixmap>
#include <QWheelEvent>
#include <QtMath>

namespace {
QColor colorFromInt(int value) {
    return QColor((value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF);
}

void drawRoundedLabel(QPainter* painter, const QRect& rect, const QColor& color, const QString& text) {
    painter->setPen(Qt::NoPen);
    painter->setBrush(QColor(10, 14, 20, 210));
    painter->drawRoundedRect(rect, 5, 5);

    painter->setBrush(color);
    painter->drawRoundedRect(QRect(rect.left(), rect.top(), 4, rect.height()), 2, 2);

    painter->setPen(QColor(235, 240, 248));
    painter->drawText(rect.adjusted(9, 0, -5, 0), Qt::AlignVCenter | Qt::AlignLeft, text);
}
}

ImageAnnotateWidget::ImageAnnotateWidget(QWidget* parent) : QWidget(parent) {
    setMouseTracking(true);
    setMinimumSize(520, 380);
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
    emitViewportStatus();
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

void ImageAnnotateWidget::setShowAnnotationLabels(bool show) {
    if (m_showAnnotationLabels == show) {
        return;
    }
    m_showAnnotationLabels = show;
    update();
}

QString ImageAnnotateWidget::viewportStatusText() const {
    if (m_image.isNull()) {
        return QString::fromUtf8(u8"未加载图像");
    }
    return QString::fromUtf8(u8"图像: %1 x %2 | 缩放: %3% | 标注: %4")
        .arg(m_image.width())
        .arg(m_image.height())
        .arg(qRound(m_zoomFactor * 100.0))
        .arg(m_annotations.size());
}

void ImageAnnotateWidget::emitViewportStatus() {
    emit viewportStatusChanged(viewportStatusText());
}

void ImageAnnotateWidget::paintEvent(QPaintEvent* event) {
    Q_UNUSED(event);

    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);
    drawCanvasBackground(&painter);

    if (m_image.isNull()) {
        drawEmptyState(&painter);
        return;
    }

    const QRect display = imageDisplayRect();
    const QRect shadowRect = display.adjusted(-8, -8, 8, 8);
    QPainterPath shadowPath;
    shadowPath.addRoundedRect(QRectF(shadowRect), 10, 10);
    painter.fillPath(shadowPath, QColor(0, 0, 0, 65));

    painter.setPen(QPen(QColor(63, 78, 102), 1.0));
    painter.setBrush(QColor(8, 11, 16));
    painter.drawRoundedRect(display.adjusted(-1, -1, 1, 1), 4, 4);
    painter.drawImage(display, m_image);

    for (int i = 0; i < m_annotations.size(); ++i) {
        const bool selected = (i == m_selectedAnnotationIndex);
        const AnnotationObject& ann = m_annotations.at(i);
        const QColor baseColor = colorFromInt(ann.colorValue);
        const QColor drawColor = selected ? QColor(255, 214, 74) : baseColor;
        const QPolygonF widgetPoly = imageToWidgetPolygon(ann.rectPolygonImage);

        painter.setPen(QPen(drawColor, selected ? 3.0 : 2.0));
        QColor fillColor = baseColor;
        fillColor.setAlpha(selected ? 58 : 30);
        painter.setBrush(fillColor);
        painter.drawPolygon(widgetPoly);

        const QRectF box = widgetPoly.boundingRect();
        if (selected) {
            painter.setPen(QPen(QColor(255, 214, 74, 210), 1.5));
            painter.setBrush(QColor(255, 214, 74));
            const QVector<QPointF> handles = {
                box.topLeft(), box.topRight(), box.bottomLeft(), box.bottomRight()
            };
            for (const QPointF& p : handles) {
                painter.drawRoundedRect(QRectF(p.x() - 4.0, p.y() - 4.0, 8.0, 8.0), 2, 2);
            }
        }

        if (m_showAnnotationLabels) {
            const QString tag = QStringLiteral("%1  %2").arg(i + 1).arg(ann.label);
            const QFontMetrics fm(painter.font());
            const QRect textBounds = fm.boundingRect(tag).adjusted(-10, -4, 10, 5);
            QPoint textPos(static_cast<int>(qRound(box.left())), static_cast<int>(qRound(box.top())) - 5);
            if (textPos.y() - textBounds.height() < 0) {
                textPos.setY(static_cast<int>(qRound(box.top())) + textBounds.height() + 6);
            }
            if (textPos.x() + textBounds.width() > width()) {
                textPos.setX(qMax(0, width() - textBounds.width() - 2));
            }

            const QRect bgRect(textPos.x(), textPos.y() - textBounds.height(), textBounds.width(), textBounds.height());
            drawRoundedLabel(&painter, bgRect, drawColor, tag);
        }
    }

    if (m_tempResult.valid) {
        if (!m_tempResult.contourImage.isEmpty()) {
            painter.setPen(QPen(QColor(255, 96, 112), 2.0));
            painter.setBrush(QColor(255, 96, 112, 45));
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
        painter.setPen(QPen(QColor(8, 11, 16), 2.0));
        painter.setBrush(QColor(98, 210, 162));
        painter.drawEllipse(p, 5.0, 5.0);
    }

    if (m_draggingRect && m_mousePressed) {
        QRect r(m_pressWidgetPos, m_currentWidgetPos);
        r = r.normalized();
        painter.setPen(QPen(QColor(77, 141, 255), 2.0, Qt::DashLine));
        painter.setBrush(QColor(77, 141, 255, 36));
        painter.drawRoundedRect(r, 3, 3);
    } else if (m_tempResult.hasRect && m_tempResult.promptRectImage.isValid()) {
        const QPointF p1 = imageToWidget(m_tempResult.promptRectImage.topLeft());
        const QPointF p2 = imageToWidget(m_tempResult.promptRectImage.bottomRight());
        QRectF wr(p1, p2);
        wr = wr.normalized();
        painter.setPen(QPen(QColor(77, 141, 255), 2.0, Qt::DashLine));
        painter.setBrush(Qt::NoBrush);
        painter.drawRoundedRect(wr, 3, 3);
    }

    if (rect().contains(m_currentWidgetPos)) {
        painter.setPen(QPen(QColor(143, 183, 255, 70), 1.0));
        painter.drawLine(QPoint(0, m_currentWidgetPos.y()), QPoint(width(), m_currentWidgetPos.y()));
        painter.drawLine(QPoint(m_currentWidgetPos.x(), 0), QPoint(m_currentWidgetPos.x(), height()));
    }

    drawHud(&painter, display);
}

void ImageAnnotateWidget::drawEmptyState(QPainter* painter) {
    const QPixmap emptyArt(QStringLiteral(":/assets/images/empty-workspace.svg"));
    const QSize artSize = emptyArt.isNull() ? QSize(0, 0) : emptyArt.size().scaled(260, 180, Qt::KeepAspectRatio);
    const int centerY = height() / 2 - 20;
    if (!emptyArt.isNull()) {
        const QRect artRect((width() - artSize.width()) / 2, centerY - artSize.height(), artSize.width(), artSize.height());
        painter->drawPixmap(artRect, emptyArt);
    }

    painter->setPen(QColor(218, 226, 240));
    QFont titleFont = painter->font();
    titleFont.setPointSize(titleFont.pointSize() + 2);
    titleFont.setBold(true);
    painter->setFont(titleFont);
    painter->drawText(QRect(24, centerY + 12, width() - 48, 30), Qt::AlignCenter,
                      QString::fromUtf8(u8"打开图像文件夹开始标注"));

    QFont hintFont = painter->font();
    hintFont.setPointSize(qMax(9, hintFont.pointSize() - 2));
    hintFont.setBold(false);
    painter->setFont(hintFont);
    painter->setPen(QColor(135, 147, 168));
    painter->drawText(QRect(24, centerY + 46, width() - 48, 28), Qt::AlignCenter,
                      QString::fromUtf8(u8"左键点击用于自动提示，拖拽可创建矩形标注，滚轮缩放，右键重置视图"));
}

void ImageAnnotateWidget::drawCanvasBackground(QPainter* painter) {
    QLinearGradient bg(0, 0, 0, height());
    bg.setColorAt(0.0, QColor(13, 17, 24));
    bg.setColorAt(1.0, QColor(9, 12, 18));
    painter->fillRect(rect(), bg);

    painter->setPen(QPen(QColor(42, 51, 66, 85), 1.0));
    constexpr int grid = 32;
    for (int x = 0; x < width(); x += grid) {
        painter->drawLine(x, 0, x, height());
    }
    for (int y = 0; y < height(); y += grid) {
        painter->drawLine(0, y, width(), y);
    }
}

void ImageAnnotateWidget::drawHud(QPainter* painter, const QRect& display) {
    const QString hud = viewportStatusText();
    const QFontMetrics fm(painter->font());
    const QRect textRect = fm.boundingRect(hud).adjusted(-10, -5, 10, 6);
    const QRect hudRect(14, 14, qMin(width() - 28, textRect.width()), textRect.height());

    painter->setPen(Qt::NoPen);
    painter->setBrush(QColor(10, 14, 20, 190));
    painter->drawRoundedRect(hudRect, 6, 6);
    painter->setPen(QColor(197, 210, 229));
    painter->drawText(hudRect.adjusted(8, 0, -8, 0), Qt::AlignVCenter | Qt::AlignLeft, hud);

    if (!display.isEmpty()) {
        const QString sizeText = QStringLiteral("%1 x %2").arg(m_image.width()).arg(m_image.height());
        const QRect sizeBounds = fm.boundingRect(sizeText).adjusted(-10, -5, 10, 6);
        const QRect sizeRect(display.right() - sizeBounds.width() - 10, display.bottom() - sizeBounds.height() - 10,
                             sizeBounds.width(), sizeBounds.height());
        painter->setPen(Qt::NoPen);
        painter->setBrush(QColor(10, 14, 20, 175));
        painter->drawRoundedRect(sizeRect, 6, 6);
        painter->setPen(QColor(197, 210, 229));
        painter->drawText(sizeRect.adjusted(8, 0, -8, 0), Qt::AlignVCenter | Qt::AlignLeft, sizeText);
    }
}

void ImageAnnotateWidget::mousePressEvent(QMouseEvent* event) {
    if (event->button() == Qt::RightButton && !m_image.isNull()) {
        m_zoomFactor = 1.0;
        m_viewOffsetWidget = QPointF(0.0, 0.0);
        update();
        emitViewportStatus();
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
    emitViewportStatus();
    event->accept();
}

void ImageAnnotateWidget::mouseMoveEvent(QMouseEvent* event) {
    m_currentWidgetPos = event->pos();

    if (!m_mousePressed || m_image.isNull()) {
        update();
        QWidget::mouseMoveEvent(event);
        return;
    }

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
