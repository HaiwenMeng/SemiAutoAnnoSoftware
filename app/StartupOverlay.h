#ifndef AUTOLABELPROJECT_APP_STARTUPOVERLAY_H
#define AUTOLABELPROJECT_APP_STARTUPOVERLAY_H

#include <QWidget>

class QTimer;

class StartupOverlay : public QWidget {
    Q_OBJECT
public:
    explicit StartupOverlay(QWidget* parent = nullptr);
    ~StartupOverlay() override;

    void showMessage(const QString& title, const QString& detail);
    void hideOverlay();

protected:
    void paintEvent(QPaintEvent* event) override;

private slots:
    void onAnimationTick();

private:
    QString m_title;
    QString m_detail;
    QTimer* m_timer = nullptr;
    int m_angle = 0;
};

#endif // AUTOLABELPROJECT_APP_STARTUPOVERLAY_H
