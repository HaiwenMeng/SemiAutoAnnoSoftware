#include <QApplication>
#include <QIcon>

#include "app/MainWindow.h"
#include "app/UiTheme.h"

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);
    app.setWindowIcon(QIcon(QStringLiteral(":/assets/icons/app.svg")));
    UiTheme::apply(&app);

    MainWindow window;
    window.show();

    return app.exec();
}
