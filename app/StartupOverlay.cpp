#include "app/StartupOverlay.h"

#include <QFont>
#include <QFontMetrics>
#include <QPainter>
#include <QPainterPath>
#include <QPaintEvent>
#include <QTimer>

StartupOverlay::StartupOverlay(QWidget* parent) : QWidget(parent), m_timer(new QTimer(this)) {
    setAttribute(Qt::WA_TransparentForMouseEvents, true);
    setAutoFillBackground(false);
    hide();

    connect(m_timer, &QTimer::timeout, this, &StartupOverlay::onAnimationTick);
}

StartupOverlay::~StartupOverlay() = default;

void StartupOverlay::showMessage(const QString& title, const QString& detail) {
    m_title = title;
    m_detail = detail;
    if (parentWidget()) {
        setGeometry(parentWidget()->rect());
    }
    raise();
    show();
    m_timer->start(24);
    update();
}

void StartupOverlay::hideOverlay() {
    m_timer->stop();
    hide();
}

void StartupOverlay::paintEvent(QPaintEvent* event) {
    Q_UNUSED(event);

    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.fillRect(rect(), QColor(8, 11, 16, 168));

    const QSize panelSize(360, 168);
    const QRect panel((width() - panelSize.width()) / 2, (height() - panelSize.height()) / 2,
                      panelSize.width(), panelSize.height());
    QPainterPath panelPath;
    panelPath.addRoundedRect(QRectF(panel), 12, 12);
    painter.fillPath(panelPath, QColor(19, 25, 35, 238));
    painter.setPen(QPen(QColor(50, 64, 86), 1.0));
    painter.drawPath(panelPath);

    const QPoint spinnerCenter(panel.center().x(), panel.top() + 52);
    const int radius = 22;
    painter.setPen(QPen(QColor(54, 67, 90), 4.0, Qt::SolidLine, Qt::RoundCap));
    painter.drawEllipse(spinnerCenter, radius, radius);
    painter.setPen(QPen(QColor(98, 210, 162), 4.0, Qt::SolidLine, Qt::RoundCap));
    painter.drawArc(QRect(spinnerCenter.x() - radius, spinnerCenter.y() - radius, radius * 2, radius * 2),
                    m_angle * 16, 110 * 16);

    QFont titleFont = painter.font();
    titleFont.setBold(true);
    titleFont.setPointSize(titleFont.pointSize() + 2);
    painter.setFont(titleFont);
    painter.setPen(QColor(233, 239, 249));
    painter.drawText(QRect(panel.left() + 24, panel.top() + 86, panel.width() - 48, 28),
                     Qt::AlignCenter, m_title);

    QFont detailFont = painter.font();
    detailFont.setBold(false);
    detailFont.setPointSize(qMax(9, detailFont.pointSize() - 2));
    painter.setFont(detailFont);
    painter.setPen(QColor(143, 155, 176));
    painter.drawText(QRect(panel.left() + 24, panel.top() + 118, panel.width() - 48, 28),
                     Qt::AlignCenter, m_detail);
}

void StartupOverlay::onAnimationTick() {
    m_angle = (m_angle + 9) % 360;
    update();
}
