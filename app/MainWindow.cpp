#include "app/MainWindow.h"

#include <exception>

#include <QCoreApplication>
#include <QDebug>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QLabel>
#include <QListWidget>
#include <QListWidgetItem>
#include <QPushButton>
#include <QSplitter>
#include <QStatusBar>
#include <QSizePolicy>
#include <QVBoxLayout>
#include <QWidget>

#include "data/AnnotationJsonIO.h"
#include "data/LabelConfigIO.h"
#include "dialogs/AddLabelDialog.h"
#include "dialogs/LabelSelectDialog.h"
#include "inference/SamInferenceBridge.h"
#include "inference/SamTypes.h"
#include "widgets/ImageAnnotateWidget.h"

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
}

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent), m_bridge(new SamInferenceBridge()) {
    setupUi();

    m_workingDir = QCoreApplication::applicationDirPath();
    setWorkingDirectory(m_workingDir);

    statusBar()->showMessage(QString::fromUtf8(u8"就绪"));
}

MainWindow::~MainWindow() {
    delete m_bridge;
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

    const int row = m_annotationList->currentRow();
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

    const int row = m_annotationList->currentRow();
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
    m_imageWidget->setAnnotations(m_annotations);
    statusBar()->showMessage(QString::fromUtf8(u8"标签已添加"));
}

void MainWindow::onDeleteLabelClicked() {
    const int row = m_labelList->currentRow();
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
    m_imageWidget->setAnnotations(m_annotations);
    statusBar()->showMessage(QString::fromUtf8(u8"标签已删除"));
}

void MainWindow::onPointPromptRequested(const QPointF& imagePoint) {
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
    m_imageWidget->setSelectedAnnotationIndex(annotationIndex);
    if (annotationIndex >= 0 && annotationIndex < m_annotationList->count()) {
        m_annotationList->setCurrentRow(annotationIndex);
    }
}

void MainWindow::setupUi() {
    resize(1360, 860);

    auto* central = new QWidget(this);
    auto* rootLayout = new QVBoxLayout(central);

    auto* toolbarRow = new QHBoxLayout();
    m_initButton = new QPushButton(QString::fromUtf8(u8"初始化SAM3"), central);
    m_openFolderButton = new QPushButton(QString::fromUtf8(u8"打开文件夹"), central);
    toolbarRow->addWidget(m_initButton);
    toolbarRow->addWidget(m_openFolderButton);
    toolbarRow->addStretch(1);

    auto* splitter = new QSplitter(Qt::Horizontal, central);

    m_imageList = new QListWidget(splitter);
    m_imageList->setMinimumWidth(260);

    auto* imagePanel = new QWidget(splitter);
    auto* imageLayout = new QVBoxLayout(imagePanel);
    imageLayout->setContentsMargins(0, 0, 0, 0);
    imageLayout->setSpacing(8);
    m_imageWidget = new ImageAnnotateWidget(imagePanel);
    imageLayout->addWidget(m_imageWidget, 1);

    auto* rightPanel = new QWidget(splitter);
    auto* rightLayout = new QVBoxLayout(rightPanel);

    auto* labelTitle = new QLabel(QString::fromUtf8(u8"标签列表"), rightPanel);
    m_labelList = new QListWidget(rightPanel);
    auto* labelButtonRow = new QHBoxLayout();
    m_addLabelButton = new QPushButton(QString::fromUtf8(u8"添加标签"), rightPanel);
    m_deleteLabelButton = new QPushButton(QString::fromUtf8(u8"删除标签"), rightPanel);
    labelButtonRow->addWidget(m_addLabelButton);
    labelButtonRow->addWidget(m_deleteLabelButton);

    auto* annTitle = new QLabel(QString::fromUtf8(u8"标注列表"), rightPanel);
    m_annotationList = new QListWidget(rightPanel);
    auto* annButtonRow = new QHBoxLayout();
    m_deleteAnnotationButton = new QPushButton(QString::fromUtf8(u8"删除标注"), rightPanel);
    m_fixAnnotationButton = new QPushButton(QString::fromUtf8(u8"修正标注"), rightPanel);
    annButtonRow->addWidget(m_deleteAnnotationButton);
    annButtonRow->addWidget(m_fixAnnotationButton);

    rightLayout->addWidget(labelTitle);
    rightLayout->addWidget(m_labelList, 1);
    rightLayout->addLayout(labelButtonRow);
    rightLayout->addWidget(annTitle);
    rightLayout->addWidget(m_annotationList, 2);
    rightLayout->addLayout(annButtonRow);

    splitter->setStretchFactor(0, 2);
    splitter->setStretchFactor(1, 6);
    splitter->setStretchFactor(2, 3);

    rootLayout->addLayout(toolbarRow);
    rootLayout->addWidget(splitter, 1);

    setCentralWidget(central);

    connect(m_initButton, &QPushButton::clicked, this, &MainWindow::onInitializeBridgeClicked);
    connect(m_openFolderButton, &QPushButton::clicked, this, &MainWindow::onOpenFolderClicked);

    connect(m_addLabelButton, &QPushButton::clicked, this, &MainWindow::onAddLabelClicked);
    connect(m_deleteLabelButton, &QPushButton::clicked, this, &MainWindow::onDeleteLabelClicked);
    connect(m_deleteAnnotationButton, &QPushButton::clicked, this, &MainWindow::onDeleteAnnotationClicked);
    connect(m_fixAnnotationButton, &QPushButton::clicked, this, &MainWindow::onFixAnnotationClicked);

    connect(m_imageList, &QListWidget::currentRowChanged, this, &MainWindow::onImageSelectionChanged);

    connect(m_imageWidget, &ImageAnnotateWidget::pointPromptRequested, this, &MainWindow::onPointPromptRequested);
    connect(m_imageWidget, &ImageAnnotateWidget::rectPromptRequested, this, &MainWindow::onRectPromptRequested);
    connect(m_imageWidget, &ImageAnnotateWidget::annotationSelectionChanged,
            this, &MainWindow::onAnnotationSelectionChanged);

    connect(m_annotationList, &QListWidget::currentRowChanged, m_imageWidget,
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
        m_imageList->setCurrentRow(0);
    } else {
        m_currentImagePath.clear();
        m_annotations.clear();
        m_imageWidget->setAnnotations(m_annotations);
        m_annotationList->clear();
        m_imageWidget->clearTempResult();
    }
}

void MainWindow::refreshImageList() {
    m_imageList->clear();
    m_imageFilePaths.clear();

    if (m_workingDir.isEmpty()) {
        return;
    }

    QDir dir(m_workingDir);
    const QStringList filters = {QStringLiteral("*.jpg"), QStringLiteral("*.jpeg"), QStringLiteral("*.png"), QStringLiteral("*.bmp")};
    const QFileInfoList files = dir.entryInfoList(filters, QDir::Files | QDir::Readable | QDir::NoSymLinks, QDir::Name);

    for (const QFileInfo& fi : files) {
        m_imageFilePaths.push_back(fi.absoluteFilePath());
        m_imageList->addItem(fi.fileName());
    }
}

bool MainWindow::loadImageByPath(const QString& imagePath) {
    if (imagePath.isEmpty()) {
        return false;
    }

    if (!m_imageWidget->loadImage(imagePath)) {
        statusBar()->showMessage(QString::fromUtf8(u8"加载图像失败"));
        appendLog(QStringLiteral("[Image] failed to load %1").arg(imagePath));
        return false;
    }

    m_currentImagePath = imagePath;
    m_imageWidget->clearTempResult();

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
    m_labelList->clear();
    for (int i = 0; i < m_labelConfig.nameList.size(); ++i) {
        const QString& name = m_labelConfig.nameList.at(i);
        auto* item = new QListWidgetItem(name);
        item->setForeground(colorFromInt(i < m_labelConfig.colorDefine.size() ? m_labelConfig.colorDefine.at(i) : 0x00FF00));
        m_labelList->addItem(item);
    }
    if (m_labelList->count() > 0 && m_labelList->currentRow() < 0) {
        m_labelList->setCurrentRow(0);
    }
}

bool MainWindow::reloadAnnotationsForCurrentImage() {
    m_annotations.clear();
    m_imageWidget->setAnnotations(m_annotations);
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
    m_imageWidget->setAnnotations(m_annotations);
    refreshAnnotationList();
    return true;
}

void MainWindow::refreshAnnotationList() {
    m_annotationList->clear();
    for (int i = 0; i < m_annotations.size(); ++i) {
        const AnnotationObject& ann = m_annotations.at(i);
        auto* item = new QListWidgetItem(QStringLiteral("%1 - %2").arg(i + 1).arg(ann.label));
        item->setForeground(colorFromInt(ann.colorValue));
        m_annotationList->addItem(item);
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

QString MainWindow::currentSelectedLabel(QString* errorMessage) const {
    if (m_labelConfig.nameList.isEmpty() || m_labelList == nullptr || m_labelList->currentRow() < 0) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("No selected label for SAM3 result.");
        }
        return QString();
    }

    const int row = m_labelList->currentRow();
    if (row >= m_labelConfig.nameList.size()) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("Selected label index is out of range.");
        }
        return QString();
    }

    return m_labelConfig.nameList.at(row);
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

    m_imageWidget->clearTempResult();
    reloadAnnotationsForCurrentImage();
    statusBar()->showMessage(QString::fromUtf8(u8"SAM3已保存%1个目标").arg(newAnnotations.size()));
    appendLog(QStringLiteral("[SAM3 SaveResult] appended %1 annotations with label %2")
                  .arg(newAnnotations.size())
                  .arg(labelName));
    return true;
}
