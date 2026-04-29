#include "dialogs/LabelSelectDialog.h"

#include <QColor>
#include <QDialogButtonBox>
#include <QListWidget>
#include <QListWidgetItem>
#include <QVBoxLayout>

namespace {
QColor colorFromInt(int value) {
    return QColor((value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF);
}
}

LabelSelectDialog::LabelSelectDialog(const QStringList& labels, const QList<int>& colors, QWidget* parent)
    : QDialog(parent), m_labels(labels), m_colors(colors) {
    setWindowTitle(QString::fromUtf8(u8"选择标签"));
    resize(340, 360);

    auto* layout = new QVBoxLayout(this);
    m_list = new QListWidget(this);
    for (int i = 0; i < m_labels.size(); ++i) {
        auto* item = new QListWidgetItem(m_labels.at(i));
        const int color = (i < m_colors.size()) ? m_colors.at(i) : 0x00FF00;
        item->setForeground(colorFromInt(color));
        m_list->addItem(item);
    }
    if (m_list->count() > 0) {
        m_list->setCurrentRow(0);
    }

    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);

    layout->addWidget(m_list, 1);
    layout->addWidget(buttons);
}

int LabelSelectDialog::selectedIndex() const {
    return m_list ? m_list->currentRow() : -1;
}

QString LabelSelectDialog::selectedLabel() const {
    const int idx = selectedIndex();
    if (idx < 0 || idx >= m_labels.size()) {
        return {};
    }
    return m_labels.at(idx);
}

int LabelSelectDialog::selectedColorValue() const {
    const int idx = selectedIndex();
    if (idx < 0 || idx >= m_colors.size()) {
        return 0x00FF00;
    }
    return m_colors.at(idx);
}
