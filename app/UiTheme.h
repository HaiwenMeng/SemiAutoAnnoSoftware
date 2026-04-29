#ifndef AUTOLABELPROJECT_APP_UITHEME_H
#define AUTOLABELPROJECT_APP_UITHEME_H

class QApplication;
class QWidget;

class UiTheme {
public:
    static bool apply(QApplication* app);
    static void applyDarkTitleBar(QWidget* widget);
};

#endif // AUTOLABELPROJECT_APP_UITHEME_H
