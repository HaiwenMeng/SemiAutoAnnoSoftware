#include "dialogs/AddLabelDialog.h"

#include <QColorDialog>
#include <QDialogButtonBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QVBoxLayout>

namespace {
QString buttonStyleFromColor(int value) {
    const int r = (value >> 16) & 0xFF;
    const int g = (value >> 8) & 0xFF;
    const int b = value & 0xFF;
    return QStringLiteral("background-color: rgb(%1,%2,%3); border:1px solid #666;").arg(r).arg(g).arg(b);
}
}

AddLabelDialog::AddLabelDialog(QWidget* parent) : QDialog(parent) {
    setWindowTitle(QString::fromUtf8(u8"添加标签"));
    resize(360, 160);

    auto* root = new QVBoxLayout(this);

    auto* nameRow = new QHBoxLayout();
    nameRow->addWidget(new QLabel(QString::fromUtf8(u8"标签名称:"), this));
    m_nameEdit = new QLineEdit(this);
    nameRow->addWidget(m_nameEdit, 1);

    auto* colorRow = new QHBoxLayout();
    colorRow->addWidget(new QLabel(QString::fromUtf8(u8"颜色:"), this));
    m_colorButton = new QPushButton(QString::fromUtf8(u8"选择"), this);
    m_colorButton->setMinimumWidth(110);
    colorRow->addWidget(m_colorButton);
    colorRow->addStretch(1);

    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);

    connect(m_colorButton, &QPushButton::clicked, this, &AddLabelDialog::onChooseColorClicked);
    connect(buttons, &QDialogButtonBox::accepted, this, &AddLabelDialog::onAcceptClicked);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);

    root->addLayout(nameRow);
    root->addLayout(colorRow);
    root->addStretch(1);
    root->addWidget(buttons);

    updateColorPreview();
}

QString AddLabelDialog::labelName() const {
    return m_nameEdit ? m_nameEdit->text().trimmed() : QString();
}

int AddLabelDialog::colorValue() const {
    return m_colorValue;
}

void AddLabelDialog::onChooseColorClicked() {
    const QColor current((m_colorValue >> 16) & 0xFF, (m_colorValue >> 8) & 0xFF, m_colorValue & 0xFF);
    const QColor color = QColorDialog::getColor(current, this, QString::fromUtf8(u8"选择标签颜色"));
    if (!color.isValid()) {
        return;
    }

    m_colorValue = (color.red() << 16) | (color.green() << 8) | color.blue();
    updateColorPreview();
}

void AddLabelDialog::onAcceptClicked() {
    if (labelName().isEmpty()) {
        QMessageBox::warning(this, QString::fromUtf8(u8"名称无效"), QString::fromUtf8(u8"标签名称不能为空。"));
        return;
    }
    accept();
}

void AddLabelDialog::updateColorPreview() {
    if (m_colorButton) {
        m_colorButton->setStyleSheet(buttonStyleFromColor(m_colorValue));
    }
}
