#ifndef AUTOLABELPROJECT_DIALOGS_ADDLABELDIALOG_H
#define AUTOLABELPROJECT_DIALOGS_ADDLABELDIALOG_H

#include <QDialog>
#include <QString>

class QLineEdit;
class QPushButton;

class AddLabelDialog : public QDialog {
    Q_OBJECT
public:
    explicit AddLabelDialog(QWidget* parent = nullptr);

    QString labelName() const;
    int colorValue() const;

private slots:
    void onChooseColorClicked();
    void onAcceptClicked();

private:
    void updateColorPreview();

    QLineEdit* m_nameEdit = nullptr;
    QPushButton* m_colorButton = nullptr;
    int m_colorValue = 0x00FF00;
};

#endif // AUTOLABELPROJECT_DIALOGS_ADDLABELDIALOG_H