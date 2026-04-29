#include "app/MainWindow.h"

#include <exception>

#include <QCoreApplication>
#include <QDebug>
#include <QDesktopServices>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QListWidget>
#include <QListWidgetItem>
#include <QMessageBox>
#include <QStatusBar>
#include <QUrl>
#include <QFile>

#include "data/AnnotationJsonIO.h"
#include "data/LabelConfigIO.h"
#include "dialogs/AddLabelDialog.h"
#include "dialogs/LabelSelectDialog.h"
#include "inference/SamInferenceBridge.h"
#include "inference/SamTypes.h"
#include "ui_MainWindow.h"

namespace {
QPolygonF toAxisAlignedRectPolygon(const QPolygonF& poly) {
    if (poly.isEmpty()) {
        return {};
    }

    qreal minX = poly.first().x();
    qreal minY = poly.first().y();
    qreal maxX = minX;
    qreal maxY = minY;
    for (const QPointF& p : poly) {
        minX = qMin(minX, p.x());
        minY = qMin(minY, p.y());
        maxX = qMax(maxX, p.x());
        maxY = qMax(maxY, p.y());
    }

    QPolygonF rect;
    rect << QPointF(minX, minY) << QPointF(maxX, minY) << QPointF(maxX, maxY) << QPointF(minX, maxY);
    return rect;
}

QColor colorFromInt(int value) {
    return QColor((value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF);
}

QPolygonF polygonFromRoiData(const QVector<double>& roiData) {
    QPolygonF poly;
    if (roiData.size() < 8) {
        return poly;
    }

    poly.reserve(4);
    for (int i = 0; i < 8; i += 2) {
        poly << QPointF(roiData.at(i), roiData.at(i + 1));
    }
    return toAxisAlignedRectPolygon(poly);
}

QString jsonPathFromImagePath(const QString& imagePath) {
    const QFileInfo info(imagePath);
    return info.absolutePath() + QLatin1Char('/') + info.completeBaseName() + QStringLiteral(".json");
}
}

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent), ui(new Ui::MainWindow), m_bridge(new SamInferenceBridge()) {
    ui->setupUi(this);
    setupConnections();

    m_workingDir = QCoreApplication::applicationDirPath();
    setWorkingDirectory(m_workingDir);

    statusBar()->showMessage(QString::fromUtf8(u8"就绪"));
}

MainWindow::~MainWindow() {
    delete m_bridge;
    delete ui;
}

void MainWindow::onInitializeBridgeClicked() {
    QString error;
    if (!m_bridge->initialize(&error)) {
        statusBar()->showMessage(QString::fromUtf8(u8"初始化失败"));
        appendLog(QStringLiteral("[Init] %1").arg(error));
        return;
    }

    statusBar()->showMessage(QString::fromUtf8(u8"SAM3初始化成功"));
    appendLog(QStringLiteral("[Init] SAM3 initialized"));

    if (!m_currentImagePath.isEmpty()) {
        if (!m_bridge->setCurrentImage(m_currentImagePath, &error)) {
            appendLog(QStringLiteral("[SetCurrentImage] %1").arg(error));
        }
    }
}

void MainWindow::onOpenFolderClicked() {
    const QString dir = QFileDialog::getExistingDirectory(this, QString::fromUtf8(u8"打开文件夹"), m_workingDir);
    if (dir.isEmpty()) {
        return;
    }

    setWorkingDirectory(dir);
}

void MainWindow::onImageSelectionChanged(int row) {
    if (row < 0 || row >= m_imageFilePaths.size()) {
        return;
    }

    loadImageByPath(m_imageFilePaths.at(row));
}

void MainWindow::onDeleteAnnotationClicked() {
    if (m_currentImagePath.isEmpty()) {
        statusBar()->showMessage(QString::fromUtf8(u8"未选择图像"));
        return;
    }

    const int row = ui->annotationList->currentRow();
    if (row < 0 || row >= m_annotations.size()) {
        statusBar()->showMessage(QString::fromUtf8(u8"请选择要删除的标注"));
        return;
    }

    const int shapeIndex = m_annotations.at(row).shapeIndex;
    QString error;
    if (!AnnotationJsonIO::removeAnnotationByIndex(m_currentImagePath, shapeIndex, &error)) {
        statusBar()->showMessage(QString::fromUtf8(u8"删除标注失败"));
        appendLog(QStringLiteral("[RemoveAnnotation] %1").arg(error));
        return;
    }

    reloadAnnotationsForCurrentImage();
    statusBar()->showMessage(QString::fromUtf8(u8"标注已删除"));
}

void MainWindow::onFixAnnotationClicked() {
    if (m_currentImagePath.isEmpty()) {
        statusBar()->showMessage(QString::fromUtf8(u8"未选择图像"));
        return;
    }

    const int row = ui->annotationList->currentRow();
    if (row < 0 || row >= m_annotations.size()) {
        statusBar()->showMessage(QString::fromUtf8(u8"请选择要修正的标注"));
        return;
    }
    if (m_labelConfig.nameList.isEmpty()) {
        statusBar()->showMessage(QString::fromUtf8(u8"没有可用标签"));
        return;
    }

    LabelSelectDialog dialog(m_labelConfig.nameList, m_labelConfig.colorDefine, this);
    if (dialog.exec() != QDialog::Accepted) {
        return;
    }

    AnnotationObject ann = m_annotations.at(row);
    ann.label = dialog.selectedLabel();
    ann.colorValue = dialog.selectedColorValue();
    ann.rectPolygonImage = toAxisAlignedRectPolygon(ann.rectPolygonImage);

    QString error;
    if (!AnnotationJsonIO::updateAnnotationByIndex(m_currentImagePath, ann.shapeIndex, ann, &error)) {
        statusBar()->showMessage(QString::fromUtf8(u8"修正标注失败"));
        appendLog(QStringLiteral("[UpdateAnnotation] %1").arg(error));
        return;
    }

    reloadAnnotationsForCurrentImage();
    statusBar()->showMessage(QString::fromUtf8(u8"标注已修正"));
}

void MainWindow::onAddLabelClicked() {
    AddLabelDialog dialog(this);
    if (dialog.exec() != QDialog::Accepted) {
        return;
    }

    QString error;
    if (!LabelConfigIO::addLabel(&m_labelConfig, dialog.labelName(), dialog.colorValue(), &error)) {
        statusBar()->showMessage(QString::fromUtf8(u8"添加标签失败"));
        appendLog(QStringLiteral("[AddLabel] %1").arg(error));
        return;
    }

    if (!saveLabelConfig()) {
        return;
    }

    refreshLabelList();
    updateAnnotationColors();
    refreshAnnotationList();
    ui->imageWidget->setAnnotations(m_annotations);
    statusBar()->showMessage(QString::fromUtf8(u8"标签已添加"));
}

void MainWindow::onDeleteLabelClicked() {
    const int row = ui->labelList->currentRow();
    QString error;
    if (!LabelConfigIO::removeLabel(&m_labelConfig, row, &error)) {
        statusBar()->showMessage(QString::fromUtf8(u8"删除标签失败"));
        appendLog(QStringLiteral("[DeleteLabel] %1").arg(error));
        return;
    }

    if (!saveLabelConfig()) {
        return;
    }

    refreshLabelList();
    updateAnnotationColors();
    refreshAnnotationList();
    ui->imageWidget->setAnnotations(m_annotations);
    statusBar()->showMessage(QString::fromUtf8(u8"标签已删除"));
}

void MainWindow::onOpenCurrentFolderClicked() {
    const QString folderPath = m_currentImagePath.isEmpty()
        ? m_workingDir
        : QFileInfo(m_currentImagePath).absolutePath();
    if (folderPath.isEmpty() || !QDir(folderPath).exists()) {
        statusBar()->showMessage(QString::fromUtf8(u8"当前图像文件夹不存在"));
        appendLog(QStringLiteral("[OpenCurrentFolder] invalid folder: %1").arg(folderPath));
        return;
    }

    if (!QDesktopServices::openUrl(QUrl::fromLocalFile(folderPath))) {
        statusBar()->showMessage(QString::fromUtf8(u8"打开当前图像文件夹失败"));
        appendLog(QStringLiteral("[OpenCurrentFolder] failed: %1").arg(folderPath));
    }
}

void MainWindow::onClearAllAnnotationsClicked() {
    if (m_currentImagePath.isEmpty()) {
        statusBar()->showMessage(QString::fromUtf8(u8"未选择图像"));
        return;
    }

    QString error;
    if (!AnnotationJsonIO::clearAnnotations(m_currentImagePath, &error)) {
        statusBar()->showMessage(QString::fromUtf8(u8"清空标注失败"));
        appendLog(QStringLiteral("[ClearAllAnnotations] %1").arg(error));
        return;
    }

    reloadAnnotationsForCurrentImage();
    statusBar()->showMessage(QString::fromUtf8(u8"当前图像标注已清空"));
}

void MainWindow::onFirstImageClicked() {
    if (m_imageFilePaths.isEmpty()) {
        statusBar()->showMessage(QString::fromUtf8(u8"没有可切换的图像"));
        return;
    }
    ui->imageList->setCurrentRow(0);
}

void MainWindow::onPreviousImageClicked() {
    if (m_imageFilePaths.isEmpty()) {
        statusBar()->showMessage(QString::fromUtf8(u8"没有可切换的图像"));
        return;
    }

    const int row = ui->imageList->currentRow();
    if (row <= 0) {
        ui->imageList->setCurrentRow(0);
        statusBar()->showMessage(QString::fromUtf8(u8"已经是第一张图"));
        return;
    }
    ui->imageList->setCurrentRow(row - 1);
}

void MainWindow::onDeleteCurrentImageClicked() {
    if (m_currentImagePath.isEmpty()) {
        statusBar()->showMessage(QString::fromUtf8(u8"未选择图像"));
        return;
    }

    const QString imagePath = m_currentImagePath;
    const QString jsonPath = jsonPathFromImagePath(imagePath);
    const QString confirmText = QString::fromUtf8(u8"确定要删除当前图像和对应 JSON 吗？\n\n%1\n%2")
                                    .arg(imagePath, jsonPath);
    const QMessageBox::StandardButton reply =
        QMessageBox::question(this, QString::fromUtf8(u8"确认删除图像"),
                              confirmText,
                              QMessageBox::Yes | QMessageBox::No,
                              QMessageBox::No);
    if (reply != QMessageBox::Yes) {
        statusBar()->showMessage(QString::fromUtf8(u8"已取消删除图像"));
        return;
    }

    const int oldRow = ui->imageList->currentRow();
    if (!QFile::remove(imagePath)) {
        statusBar()->showMessage(QString::fromUtf8(u8"删除图像失败"));
        appendLog(QStringLiteral("[DeleteImage] failed to delete image: %1").arg(imagePath));
        return;
    }

    bool jsonDeleteFailed = false;
    if (QFileInfo::exists(jsonPath) && !QFile::remove(jsonPath)) {
        jsonDeleteFailed = true;
        appendLog(QStringLiteral("[DeleteImage] failed to delete json: %1").arg(jsonPath));
    }

    refreshImageList();
    if (m_imageFilePaths.isEmpty()) {
        clearCurrentImageState();
        statusBar()->showMessage(jsonDeleteFailed
            ? QString::fromUtf8(u8"图像已删除，但对应JSON删除失败")
            : QString::fromUtf8(u8"图像已删除，当前文件夹无图像"));
        return;
    }

    const int nextRow = qBound(0, oldRow, m_imageFilePaths.size() - 1);
    ui->imageList->setCurrentRow(nextRow);
    statusBar()->showMessage(jsonDeleteFailed
        ? QString::fromUtf8(u8"图像已删除，但对应JSON删除失败")
        : QString::fromUtf8(u8"图像已删除"));
}

void MainWindow::onNextImageClicked() {
    if (m_imageFilePaths.isEmpty()) {
        statusBar()->showMessage(QString::fromUtf8(u8"没有可切换的图像"));
        return;
    }

    const int row = ui->imageList->currentRow();
    if (row < 0) {
        ui->imageList->setCurrentRow(0);
        return;
    }
    if (row >= m_imageFilePaths.size() - 1) {
        ui->imageList->setCurrentRow(m_imageFilePaths.size() - 1);
        statusBar()->showMessage(QString::fromUtf8(u8"已经是最后一张图"));
        return;
    }
    ui->imageList->setCurrentRow(row + 1);
}

void MainWindow::onFinalImageClicked() {
    if (m_imageFilePaths.isEmpty()) {
        statusBar()->showMessage(QString::fromUtf8(u8"没有可切换的图像"));
        return;
    }
    ui->imageList->setCurrentRow(m_imageFilePaths.size() - 1);
}

void MainWindow::onPointPromptRequested(const QPointF& imagePoint) {
    if (!isAutoAnnotationMode()) {
        Q_UNUSED(imagePoint);
        statusBar()->showMessage(QString::fromUtf8(u8"手动模式请拖拽矩形标注"));
        return;
    }

    if (!m_bridge->isInitialized()) {
        statusBar()->showMessage(QString::fromUtf8(u8"推理模型未初始化"));
        return;
    }

    QString labelError;
    const QString labelName = currentSelectedLabel(&labelError);
    if (labelName.isEmpty()) {
        statusBar()->showMessage(QString::fromUtf8(u8"未选择标签"));
        appendLog(QStringLiteral("[InferByPoint] %1").arg(labelError));
        return;
    }

    SamInferResult result;
    try {
        result = m_bridge->inferByPoint(imagePoint, labelName);
    } catch (const std::exception& ex) {
        statusBar()->showMessage(QString::fromUtf8(u8"推理异常"));
        appendLog(QStringLiteral("[InferByPoint] exception: %1").arg(ex.what()));
        return;
    } catch (...) {
        statusBar()->showMessage(QString::fromUtf8(u8"推理异常"));
        appendLog(QStringLiteral("[InferByPoint] unknown exception"));
        return;
    }

    if (!result.success) {
        statusBar()->showMessage(QString::fromUtf8(u8"推理失败"));
        appendLog(QStringLiteral("[InferByPoint] %1").arg(result.errorMessage));
        return;
    }

    if (!saveSamResultAnnotations(result, labelName)) {
        return;
    }
}

void MainWindow::onRectPromptRequested(const QRectF& imageRect) {
    if (!isAutoAnnotationMode()) {
        QString labelError;
        const QString labelName = currentSelectedLabel(&labelError);
        if (labelName.isEmpty()) {
            statusBar()->showMessage(QString::fromUtf8(u8"未选择标签"));
            appendLog(QStringLiteral("[ManualRect] %1").arg(labelError));
            return;
        }
        saveManualRectAnnotation(imageRect, labelName);
        return;
    }

    if (!m_bridge->isInitialized()) {
        statusBar()->showMessage(QString::fromUtf8(u8"推理模型未初始化"));
        return;
    }

    QString labelError;
    const QString labelName = currentSelectedLabel(&labelError);
    if (labelName.isEmpty()) {
        statusBar()->showMessage(QString::fromUtf8(u8"未选择标签"));
        appendLog(QStringLiteral("[InferByRect] %1").arg(labelError));
        return;
    }

    SamInferResult result;
    try {
        result = m_bridge->inferByRect(imageRect, labelName);
    } catch (const std::exception& ex) {
        statusBar()->showMessage(QString::fromUtf8(u8"推理异常"));
        appendLog(QStringLiteral("[InferByRect] exception: %1").arg(ex.what()));
        return;
    } catch (...) {
        statusBar()->showMessage(QString::fromUtf8(u8"推理异常"));
        appendLog(QStringLiteral("[InferByRect] unknown exception"));
        return;
    }

    if (!result.success) {
        statusBar()->showMessage(QString::fromUtf8(u8"推理失败"));
        appendLog(QStringLiteral("[InferByRect] %1").arg(result.errorMessage));
        return;
    }

    if (!saveSamResultAnnotations(result, labelName)) {
        return;
    }
}

void MainWindow::onAnnotationSelectionChanged(int annotationIndex) {
    ui->imageWidget->setSelectedAnnotationIndex(annotationIndex);
    if (annotationIndex >= 0 && annotationIndex < ui->annotationList->count()) {
        ui->annotationList->setCurrentRow(annotationIndex);
    }
}

void MainWindow::setupConnections() {

    connect(ui->initButton, &QPushButton::clicked, this, &MainWindow::onInitializeBridgeClicked);
    connect(ui->openFolderButton, &QPushButton::clicked, this, &MainWindow::onOpenFolderClicked);
    connect(ui->pushButton_5, &QPushButton::clicked, this, &MainWindow::onOpenCurrentFolderClicked);
    connect(ui->pushButton_clearAllAnno, &QPushButton::clicked, this, &MainWindow::onClearAllAnnotationsClicked);
    connect(ui->pushButton_firstImg, &QPushButton::clicked, this, &MainWindow::onFirstImageClicked);
    connect(ui->PB_LastImg, &QPushButton::clicked, this, &MainWindow::onPreviousImageClicked);
    connect(ui->pushButton_deleteImg, &QPushButton::clicked, this, &MainWindow::onDeleteCurrentImageClicked);
    connect(ui->pushButton_nextImg, &QPushButton::clicked, this, &MainWindow::onNextImageClicked);
    connect(ui->pushButton_finalImg, &QPushButton::clicked, this, &MainWindow::onFinalImageClicked);

    connect(ui->addLabelButton, &QPushButton::clicked, this, &MainWindow::onAddLabelClicked);
    connect(ui->deleteLabelButton, &QPushButton::clicked, this, &MainWindow::onDeleteLabelClicked);
    connect(ui->deleteAnnotationButton, &QPushButton::clicked, this, &MainWindow::onDeleteAnnotationClicked);
    connect(ui->fixAnnotationButton, &QPushButton::clicked, this, &MainWindow::onFixAnnotationClicked);

    connect(ui->imageList, &QListWidget::currentRowChanged, this, &MainWindow::onImageSelectionChanged);

    connect(ui->imageWidget, &ImageAnnotateWidget::pointPromptRequested, this, &MainWindow::onPointPromptRequested);
    connect(ui->imageWidget, &ImageAnnotateWidget::rectPromptRequested, this, &MainWindow::onRectPromptRequested);
    connect(ui->imageWidget, &ImageAnnotateWidget::annotationSelectionChanged,
            this, &MainWindow::onAnnotationSelectionChanged);

    connect(ui->annotationList, &QListWidget::currentRowChanged, ui->imageWidget,
            &ImageAnnotateWidget::setSelectedAnnotationIndex);
}

void MainWindow::appendLog(const QString& message) {
    qDebug().noquote() << message;
}

void MainWindow::updateWindowTitle() {
    if (m_workingDir.isEmpty()) {
        setWindowTitle(QString::fromUtf8(u8"颖图半自动标注软件"));
    } else {
        setWindowTitle(QString::fromUtf8(u8"颖图半自动标注软件-[%1]").arg(m_workingDir));
    }
}

void MainWindow::setWorkingDirectory(const QString& folderPath) {
    QDir dir(folderPath);
    if (!dir.exists()) {
        return;
    }

    m_workingDir = dir.absolutePath();
    m_labelConfigPath = QDir(m_workingDir).filePath(QStringLiteral("DefineLabel.json"));

    updateWindowTitle();

    loadLabelConfig();
    refreshLabelList();

    refreshImageList();
    if (!m_imageFilePaths.isEmpty()) {
        ui->imageWidget->setVisible(true);
        ui->imageList->setCurrentRow(0);
    } else {
        clearCurrentImageState();
    }
}

void MainWindow::refreshImageList() {
    ui->imageList->clear();
    m_imageFilePaths.clear();

    if (m_workingDir.isEmpty()) {
        return;
    }

    QDir dir(m_workingDir);
    const QStringList filters = {QStringLiteral("*.jpg"), QStringLiteral("*.jpeg"), QStringLiteral("*.png"), QStringLiteral("*.bmp")};
    const QFileInfoList files = dir.entryInfoList(filters, QDir::Files | QDir::Readable | QDir::NoSymLinks, QDir::Name);

    for (const QFileInfo& fi : files) {
        m_imageFilePaths.push_back(fi.absoluteFilePath());
        ui->imageList->addItem(fi.fileName());
    }
}

bool MainWindow::loadImageByPath(const QString& imagePath) {
    if (imagePath.isEmpty()) {
        return false;
    }

    if (!ui->imageWidget->loadImage(imagePath)) {
        statusBar()->showMessage(QString::fromUtf8(u8"加载图像失败"));
        appendLog(QStringLiteral("[Image] failed to load %1").arg(imagePath));
        return false;
    }

    ui->imageWidget->setVisible(true);
    m_currentImagePath = imagePath;
    ui->imageWidget->clearTempResult();

    if (!reloadAnnotationsForCurrentImage()) {
        return false;
    }

    if (!m_bridge->isInitialized()) {
        statusBar()->showMessage(QString::fromUtf8(u8"图像已加载，推理模型未初始化"));
        return true;
    }

    QString error;
    if (!m_bridge->setCurrentImage(imagePath, &error)) {
        statusBar()->showMessage(QString::fromUtf8(u8"设置图像失败"));
        appendLog(QStringLiteral("[SetCurrentImage] %1").arg(error));
        return false;
    }

    statusBar()->showMessage(QString::fromUtf8(u8"图像已加载"));
    return true;
}

bool MainWindow::loadLabelConfig() {
    if (m_labelConfigPath.isEmpty()) {
        return false;
    }

    QString error;
    if (LabelConfigIO::loadConfig(m_labelConfigPath, &m_labelConfig, &error)) {
        if (m_labelConfig.infoSet.isEmpty()) {
            m_labelConfig.infoSet = QString::fromUtf8(u8"默认标签集");
        }
        return true;
    }

    m_labelConfig.infoSet = QString::fromUtf8(u8"默认标签集");
    m_labelConfig.nameList = QStringList{QString::fromUtf8(u8"背景")};
    m_labelConfig.colorDefine = QList<int>{0x00FF00};
    saveLabelConfig();
    appendLog(QStringLiteral("[LabelConfig] %1").arg(error));
    return false;
}

bool MainWindow::saveLabelConfig() {
    QString error;
    if (!LabelConfigIO::saveConfig(m_labelConfigPath, m_labelConfig, &error)) {
        statusBar()->showMessage(QString::fromUtf8(u8"保存标签配置失败"));
        appendLog(QStringLiteral("[LabelConfig] %1").arg(error));
        return false;
    }
    return true;
}

void MainWindow::refreshLabelList() {
    const QString previousLabel = ui->comboBox_CurrentLabel->currentText();

    ui->labelList->clear();
    ui->comboBox_CurrentLabel->clear();
    for (int i = 0; i < m_labelConfig.nameList.size(); ++i) {
        const QString& name = m_labelConfig.nameList.at(i);
        auto* item = new QListWidgetItem(name);
        item->setForeground(colorFromInt(i < m_labelConfig.colorDefine.size() ? m_labelConfig.colorDefine.at(i) : 0x00FF00));
        ui->labelList->addItem(item);
        ui->comboBox_CurrentLabel->addItem(name);
    }
    if (ui->labelList->count() > 0 && ui->labelList->currentRow() < 0) {
        ui->labelList->setCurrentRow(0);
    }

    const int previousIndex = ui->comboBox_CurrentLabel->findText(previousLabel);
    if (previousIndex >= 0) {
        ui->comboBox_CurrentLabel->setCurrentIndex(previousIndex);
    } else if (ui->comboBox_CurrentLabel->count() > 0) {
        ui->comboBox_CurrentLabel->setCurrentIndex(0);
    }
}

bool MainWindow::reloadAnnotationsForCurrentImage() {
    m_annotations.clear();
    ui->imageWidget->setAnnotations(m_annotations);
    refreshAnnotationList();

    if (m_currentImagePath.isEmpty()) {
        return true;
    }

    QString error;
    if (!AnnotationJsonIO::loadAnnotations(m_currentImagePath, &m_annotations, &error)) {
        statusBar()->showMessage(QString::fromUtf8(u8"加载标注失败"));
        appendLog(QStringLiteral("[LoadAnnotations] %1").arg(error));
        return false;
    }

    updateAnnotationColors();
    ui->imageWidget->setAnnotations(m_annotations);
    refreshAnnotationList();
    return true;
}

void MainWindow::refreshAnnotationList() {
    ui->annotationList->clear();
    for (int i = 0; i < m_annotations.size(); ++i) {
        const AnnotationObject& ann = m_annotations.at(i);
        auto* item = new QListWidgetItem(QStringLiteral("%1 - %2").arg(i + 1).arg(ann.label));
        item->setForeground(colorFromInt(ann.colorValue));
        ui->annotationList->addItem(item);
    }
}

void MainWindow::updateAnnotationColors() {
    for (AnnotationObject& ann : m_annotations) {
        ann.colorValue = colorForLabel(ann.label);
    }
}

int MainWindow::colorForLabel(const QString& label) const {
    const int idx = m_labelConfig.nameList.indexOf(label);
    if (idx >= 0 && idx < m_labelConfig.colorDefine.size()) {
        return m_labelConfig.colorDefine.at(idx);
    }
    return 0x00C8FF;
}

bool MainWindow::isAutoAnnotationMode() const {
    return ui->comboBox_AnnoMode->currentIndex() == 0;
}

void MainWindow::clearCurrentImageState() {
    m_currentImagePath.clear();
    m_annotations.clear();
    ui->imageWidget->setAnnotations(m_annotations);
    ui->imageWidget->clearTempResult();
//    ui->imageWidget->setVisible(false);
    ui->annotationList->clear();
}

bool MainWindow::saveManualRectAnnotation(const QRectF& imageRect, const QString& labelName) {
    if (m_currentImagePath.isEmpty()) {
        statusBar()->showMessage(QString::fromUtf8(u8"未选择图像"));
        return false;
    }

    const QRectF rect = imageRect.normalized();
    if (!rect.isValid() || rect.width() < 2.0 || rect.height() < 2.0) {
        statusBar()->showMessage(QString::fromUtf8(u8"手动矩形太小"));
        appendLog(QStringLiteral("[ManualRect] rectangle is too small"));
        return false;
    }

    QPolygonF rectPolygon;
    rectPolygon << rect.topLeft()
                << QPointF(rect.right(), rect.top())
                << rect.bottomRight()
                << QPointF(rect.left(), rect.bottom());

    AnnotationObject annotation;
    annotation.label = labelName;
    annotation.colorValue = colorForLabel(labelName);
    annotation.rectPolygonImage = toAxisAlignedRectPolygon(rectPolygon);

    QString error;
    if (!AnnotationJsonIO::appendAnnotation(m_currentImagePath, annotation, &error)) {
        statusBar()->showMessage(QString::fromUtf8(u8"保存手动标注失败"));
        appendLog(QStringLiteral("[ManualRect] %1").arg(error));
        return false;
    }

    ui->imageWidget->clearTempResult();
    reloadAnnotationsForCurrentImage();
    statusBar()->showMessage(QString::fromUtf8(u8"手动标注已保存"));
    return true;
}

QString MainWindow::currentSelectedLabel(QString* errorMessage) const {
    if (m_labelConfig.nameList.isEmpty() || ui->comboBox_CurrentLabel->currentIndex() < 0) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("No selected label for annotation.");
        }
        return QString();
    }

    const QString labelName = ui->comboBox_CurrentLabel->currentText();
    if (labelName.isEmpty() || !m_labelConfig.nameList.contains(labelName)) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("Selected label is not in label config.");
        }
        return QString();
    }

    return labelName;
}

QList<AnnotationObject> MainWindow::annotationsFromSamResult(const SamInferResult& result, const QString& labelName,
                                                             QString* errorMessage) {
    QList<AnnotationObject> annotations;
    if (!result.success) {
        if (errorMessage) {
            *errorMessage = result.errorMessage.isEmpty() ? QStringLiteral("SAM3 inference failed.") : result.errorMessage;
        }
        return annotations;
    }

    for (const SamObjectResult& object : result.objects) {
        if (!object.success) {
            appendLog(QStringLiteral("[SAM3 Result] skip unsuccessful object: %1").arg(object.errorMessage));
            continue;
        }

        QPolygonF rect = polygonFromRoiData(object.minRectRoiData);
        if (rect.size() != 4) {
            rect = polygonFromRoiData(object.roiData);
        }
        if (rect.size() != 4) {
            appendLog(QStringLiteral("[SAM3 Result] skip object with invalid geometry, score=%1").arg(object.score));
            continue;
        }

        AnnotationObject annotation;
        annotation.label = labelName;
        annotation.colorValue = colorForLabel(labelName);
        annotation.rectPolygonImage = rect;
        annotations.push_back(annotation);
    }

    if (annotations.isEmpty() && errorMessage) {
        *errorMessage = QStringLiteral("SAM3 returned no valid annotation geometry.");
    }
    return annotations;
}

bool MainWindow::saveSamResultAnnotations(const SamInferResult& result, const QString& labelName) {
    QString error;
    const QList<AnnotationObject> newAnnotations = annotationsFromSamResult(result, labelName, &error);
    if (newAnnotations.isEmpty()) {
        statusBar()->showMessage(QString::fromUtf8(u8"推理结果无有效目标"));
        appendLog(QStringLiteral("[SAM3 SaveResult] %1").arg(error));
        return false;
    }

    if (!AnnotationJsonIO::appendAnnotations(m_currentImagePath, newAnnotations, &error)) {
        statusBar()->showMessage(QString::fromUtf8(u8"保存SAM3标注失败"));
        appendLog(QStringLiteral("[SAM3 SaveResult] %1").arg(error));
        return false;
    }

    ui->imageWidget->clearTempResult();
    reloadAnnotationsForCurrentImage();
    statusBar()->showMessage(QString::fromUtf8(u8"SAM3已保存%1个目标").arg(newAnnotations.size()));
    appendLog(QStringLiteral("[SAM3 SaveResult] appended %1 annotations with label %2")
                  .arg(newAnnotations.size())
                  .arg(labelName));
    return true;
}
