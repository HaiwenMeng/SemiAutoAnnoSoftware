#ifndef AUTOLABELPROJECT_DIALOGS_LABELSELECTDIALOG_H
#define AUTOLABELPROJECT_DIALOGS_LABELSELECTDIALOG_H

#include <QDialog>
#include <QList>
#include <QStringList>

class QListWidget;

class LabelSelectDialog : public QDialog {
    Q_OBJECT
public:
    explicit LabelSelectDialog(const QStringList& labels, const QList<int>& colors, QWidget* parent = nullptr);

    int selectedIndex() const;
    QString selectedLabel() const;
    int selectedColorValue() const;

private:
    QListWidget* m_list = nullptr;
    QStringList m_labels;
    QList<int> m_colors;
};

#endif // AUTOLABELPROJECT_DIALOGS_LABELSELECTDIALOG_H