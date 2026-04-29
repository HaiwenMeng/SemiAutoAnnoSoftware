#include "app/UiTheme.h"

#include <QApplication>
#include <QColor>
#include <QFile>
#include <QFont>
#include <QIODevice>
#include <QPalette>
#include <QString>
#include <QWidget>

#ifdef Q_OS_WIN
#include <windows.h>
#include <dwmapi.h>
#endif

bool UiTheme::apply(QApplication* app) {
    if (app == nullptr) {
        return false;
    }

    QFont font = app->font();
    font.setFamily(QStringLiteral("Microsoft YaHei UI"));
    font.setPointSize(10);
    app->setFont(font);

    QPalette palette;
    palette.setColor(QPalette::Window, QColor(20, 24, 31));
    palette.setColor(QPalette::WindowText, QColor(229, 235, 245));
    palette.setColor(QPalette::Base, QColor(15, 18, 24));
    palette.setColor(QPalette::AlternateBase, QColor(24, 29, 38));
    palette.setColor(QPalette::Text, QColor(229, 235, 245));
    palette.setColor(QPalette::Button, QColor(37, 44, 56));
    palette.setColor(QPalette::ButtonText, QColor(229, 235, 245));
    palette.setColor(QPalette::Highlight, QColor(51, 145, 255));
    palette.setColor(QPalette::HighlightedText, QColor(255, 255, 255));
    palette.setColor(QPalette::ToolTipBase, QColor(37, 44, 56));
    palette.setColor(QPalette::ToolTipText, QColor(229, 235, 245));
    app->setPalette(palette);

    QFile file(QStringLiteral(":/styles/dark.qss"));
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return false;
    }
    app->setStyleSheet(QString::fromUtf8(file.readAll()));
    return true;
}

void UiTheme::applyDarkTitleBar(QWidget* widget) {
    if (widget == nullptr) {
        return;
    }

#ifdef Q_OS_WIN
    const HWND hwnd = reinterpret_cast<HWND>(widget->winId());
    if (hwnd == nullptr) {
        return;
    }

    const BOOL enabled = TRUE;
    constexpr DWORD dwmwaUseImmersiveDarkMode = 20;
    HRESULT hr = DwmSetWindowAttribute(hwnd, dwmwaUseImmersiveDarkMode, &enabled, sizeof(enabled));
    if (FAILED(hr)) {
        constexpr DWORD dwmwaUseImmersiveDarkModeBefore20H1 = 19;
        DwmSetWindowAttribute(hwnd, dwmwaUseImmersiveDarkModeBefore20H1, &enabled, sizeof(enabled));
    }

    SetWindowPos(hwnd, nullptr, 0, 0, 0, 0,
                 SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
    RedrawWindow(hwnd, nullptr, nullptr, RDW_INVALIDATE | RDW_FRAME);
#else
    Q_UNUSED(widget);
#endif
}
