#include "app/MainWindow.h"

#include <exception>

#include <QCheckBox>
#include <QComboBox>
#include <QCoreApplication>
#include <QDebug>
#include <QDesktopServices>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFont>
#include <QFrame>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QIcon>
#include <QLabel>
#include <QLayout>
#include <QListWidget>
#include <QListWidgetItem>
#include <QMessageBox>
#include <QMetaObject>
#include <QMetaType>
#include <QPushButton>
#include <QResizeEvent>
#include <QShowEvent>
#include <QSizePolicy>
#include <QSplitter>
#include <QStatusBar>
#include <QThread>
#include <QTimer>
#include <QUrl>
#include <QVBoxLayout>
#include <QShortcut>
#include <QSignalBlocker>

#include "app/StartupOverlay.h"
#include "app/UiTheme.h"
#include "data/AnnotationJsonIO.h"
#include "data/LabelConfigIO.h"
#include "dialogs/AddLabelDialog.h"
#include "dialogs/LabelSelectDialog.h"
#include "inference/SamInferenceWorker.h"
#include "inference/SamTypes.h"
#include "ui_MainWindow.h"

namespace {
constexpr double kNmsIouThreshold = 0.5;

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

QRectF rectFromAnnotation(const AnnotationObject& annotation) {
    return toAxisAlignedRectPolygon(annotation.rectPolygonImage).boundingRect().normalized();
}

double rectIou(const QRectF& a, const QRectF& b) {
    if (!a.isValid() || !b.isValid()) {
        return 0.0;
    }

    const QRectF intersection = a.intersected(b);
    if (!intersection.isValid() || intersection.width() <= 0.0 || intersection.height() <= 0.0) {
        return 0.0;
    }

    const double intersectionArea = intersection.width() * intersection.height();
    const double unionArea = a.width() * a.height() + b.width() * b.height() - intersectionArea;
    if (unionArea <= 0.0) {
        return 0.0;
    }
    return intersectionArea / unionArea;
}

QString jsonPathFromImagePath(const QString& imagePath) {
    const QFileInfo info(imagePath);
    return info.absolutePath() + QLatin1Char('/') + info.completeBaseName() + QStringLiteral(".json");
}

QFrame* createPanel(QWidget* parent) {
    auto* panel = new QFrame(parent);
    panel->setObjectName(QStringLiteral("panelFrame"));
    panel->setFrameShape(QFrame::NoFrame);
    return panel;
}

void markSectionTitle(QLabel* label) {
    if (label == nullptr) {
        return;
    }
    label->setProperty("class", QStringLiteral("sectionTitle"));
    QFont font = label->font();
    font.setBold(true);
    label->setFont(font);
}

void configureActionButton(QPushButton* button, const QString& text, const QString& iconPath, const QString& toolTip,
                           bool danger = false) {
    if (button == nullptr) {
        return;
    }
    button->setText(text);
    button->setToolTip(toolTip);
    button->setIcon(QIcon(iconPath));
    button->setIconSize(QSize(18, 18));
    button->setProperty("danger", danger);
}
}

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent), ui(new Ui::MainWindow) {
    qRegisterMetaType<SamInferResult>("SamInferResult");
    qRegisterMetaType<QVector<QRectF>>("QVector<QRectF>");

    ui->setupUi(this);
    setupCommercialWorkspace();
    setupStatusBarWidgets();
    setupStartupOverlay();
    setupInferenceWorker();
    applyStaticTextAndIcons();
    setupConnections();

    m_workingDir = QCoreApplication::applicationDirPath();
    setModelStatusText(QString::fromUtf8(u8"模型未初始化"));
    setImageStatusText(QString::fromUtf8(u8"未加载图像"));
    setWorkingDirectory(m_workingDir);
    statusBar()->showMessage(QString::fromUtf8(u8"就绪"));
    QTimer::singleShot(0, this, [this]() {
        startSamInitialization(true);
    });
}

MainWindow::~MainWindow() {
    if (m_inferenceThread) {
        m_inferenceThread->quit();
        m_inferenceThread->wait();
    }
    delete ui;
}

void MainWindow::setupCommercialWorkspace() {
    if (ui->centralWidget->layout()) {
        delete ui->centralWidget->layout();
    }

    auto* root = new QVBoxLayout(ui->centralWidget);
    root->setContentsMargins(10, 10, 10, 8);
    root->setSpacing(8);

    auto* commandBar = new QFrame(ui->centralWidget);
    commandBar->setObjectName(QStringLiteral("commandBar"));
    auto* commandLayout = new QHBoxLayout(commandBar);
    commandLayout->setContentsMargins(10, 7, 10, 7);
    commandLayout->setSpacing(8);

    auto* productTitle = new QLabel(QString::fromUtf8(u8"工作台"), commandBar);
    productTitle->setProperty("class", QStringLiteral("sectionTitle"));
    QFont tFont = productTitle->font();
    tFont.setPixelSize(20);
    productTitle->setFont(tFont);
    productTitle->setMinimumWidth(60);
    commandLayout->addWidget(productTitle);
    commandLayout->addSpacing(8);
    commandLayout->addWidget(ui->openFolderButton);
    commandLayout->addWidget(ui->initButton);
    commandLayout->addWidget(ui->pushButton_5);
    commandLayout->addSpacing(10);
    commandLayout->addWidget(ui->label_3);
    commandLayout->addWidget(ui->comboBox_AnnoMode);
    commandLayout->addWidget(ui->label);
    commandLayout->addWidget(ui->comboBox_CurrentLabel);
    commandLayout->addWidget(ui->label_4);
    commandLayout->addWidget(ui->checkBox_HideLabel);
    commandLayout->addStretch(1);
    root->addWidget(commandBar);

    auto* splitter = new QSplitter(Qt::Horizontal, ui->centralWidget);
    splitter->setChildrenCollapsible(false);

    auto* leftPanel = createPanel(splitter);
    auto* leftLayout = new QVBoxLayout(leftPanel);
    leftLayout->setContentsMargins(10, 10, 10, 10);
    leftLayout->setSpacing(8);
    auto* imageHeaderLayout = new QHBoxLayout();
    imageHeaderLayout->setContentsMargins(0, 0, 0, 0);
    imageHeaderLayout->setSpacing(8);
    imageHeaderLayout->addWidget(ui->label_2);
    imageHeaderLayout->addWidget(ui->comboBox_selectImg);
    leftLayout->addLayout(imageHeaderLayout);
    leftLayout->addWidget(ui->imageList, 1);

    auto* centerHost = new QWidget(splitter);
    auto* centerLayout = new QVBoxLayout(centerHost);
    centerLayout->setContentsMargins(0, 0, 0, 0);
    centerLayout->setSpacing(8);
    ui->imageWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    centerLayout->addWidget(ui->imageWidget, 1);

    auto* navigationBar = new QFrame(centerHost);
    navigationBar->setObjectName(QStringLiteral("navigationBar"));
    auto* navLayout = new QHBoxLayout(navigationBar);
    navLayout->setContentsMargins(10, 7, 10, 7);
    navLayout->setSpacing(8);
    navLayout->addStretch(1);
    navLayout->addWidget(ui->pushButton_firstImg);
    navLayout->addWidget(ui->PB_LastImg);
    navLayout->addWidget(ui->pushButton_InferByRects);
    navLayout->addWidget(ui->pushButton_ClearRects);
    navLayout->addWidget(ui->pushButton_deleteImg);
    navLayout->addWidget(ui->pushButton_nextImg);
    navLayout->addWidget(ui->pushButton_finalImg);
    navLayout->addStretch(1);
    centerLayout->addWidget(navigationBar);

    auto* rightPanel = createPanel(splitter);
    auto* rightLayout = new QVBoxLayout(rightPanel);
    rightLayout->setContentsMargins(10, 10, 10, 10);
    rightLayout->setSpacing(8);
    rightLayout->addWidget(ui->labelTitle);
    rightLayout->addWidget(ui->labelList, 1);

    auto* labelButtonRow = new QHBoxLayout();
    labelButtonRow->setSpacing(8);
    labelButtonRow->addWidget(ui->addLabelButton);
    labelButtonRow->addWidget(ui->deleteLabelButton);
    rightLayout->addLayout(labelButtonRow);

    rightLayout->addSpacing(4);
    rightLayout->addWidget(ui->annTitle);
    rightLayout->addWidget(ui->annotationList, 1);

    auto* annButtonRow = new QGridLayout();
    annButtonRow->setHorizontalSpacing(8);
    annButtonRow->setVerticalSpacing(8);
    annButtonRow->addWidget(ui->deleteAnnotationButton, 0, 0);
    annButtonRow->addWidget(ui->fixAnnotationButton, 0, 1);
    annButtonRow->addWidget(ui->pushButton_clearAllAnno, 1, 0, 1, 2);
    rightLayout->addLayout(annButtonRow);

    splitter->addWidget(leftPanel);
    splitter->addWidget(centerHost);
    splitter->addWidget(rightPanel);
    splitter->setStretchFactor(0, 0);
    splitter->setStretchFactor(1, 1);
    splitter->setStretchFactor(2, 0);
    splitter->setSizes(QList<int>() << 260 << 760 << 300);
    root->addWidget(splitter, 1);

    if (auto* oldSplitter = ui->centralWidget->findChild<QSplitter*>(QStringLiteral("splitter"))) {
        oldSplitter->hide();
    }
}

void MainWindow::setupStatusBarWidgets() {
    m_modelStatusLabel = new QLabel(this);
    m_imageStatusLabel = new QLabel(this);
    m_folderStatusLabel = new QLabel(this);
    m_modelStatusLabel->setMinimumWidth(130);
    m_imageStatusLabel->setMinimumWidth(220);
    m_folderStatusLabel->setMinimumWidth(260);
    m_modelStatusLabel->setProperty("class", QStringLiteral("metaText"));
    m_imageStatusLabel->setProperty("class", QStringLiteral("metaText"));
    m_folderStatusLabel->setProperty("class", QStringLiteral("metaText"));
    statusBar()->addPermanentWidget(m_modelStatusLabel);
    statusBar()->addPermanentWidget(m_imageStatusLabel);
    statusBar()->addPermanentWidget(m_folderStatusLabel, 1);
}

void MainWindow::setupStartupOverlay() {
    m_startupOverlay = new StartupOverlay(ui->centralWidget);
    m_startupOverlay->setGeometry(ui->centralWidget->rect());
    m_startupOverlay->hide();
}

void MainWindow::setupInferenceWorker() {
    m_inferenceThread = new QThread(this);
    m_inferenceWorker = new SamInferenceWorker();
    m_inferenceWorker->moveToThread(m_inferenceThread);

    connect(m_inferenceThread, &QThread::finished, m_inferenceWorker, &QObject::deleteLater);
    connect(m_inferenceWorker, &SamInferenceWorker::initializeFinished,
            this, &MainWindow::onSamInitializeFinished);
    connect(m_inferenceWorker, &SamInferenceWorker::currentImageFinished,
            this, &MainWindow::onSamCurrentImageFinished);
    connect(m_inferenceWorker, &SamInferenceWorker::pointInferenceFinished,
            this, &MainWindow::onSamPointInferenceFinished);
    connect(m_inferenceWorker, &SamInferenceWorker::rectInferenceFinished,
            this, &MainWindow::onSamRectInferenceFinished);
    connect(m_inferenceWorker, &SamInferenceWorker::rectsInferenceFinished,
            this, &MainWindow::onSamRectsInferenceFinished);

    m_inferenceThread->start();
}

void MainWindow::resizeEvent(QResizeEvent* event) {
    QMainWindow::resizeEvent(event);
    if (m_startupOverlay) {
        m_startupOverlay->setGeometry(ui->centralWidget->rect());
        if (m_startupOverlay->isVisible()) {
            m_startupOverlay->raise();
        }
    }
}

void MainWindow::showEvent(QShowEvent* event) {
    QMainWindow::showEvent(event);
    applyNativeTitleBarTheme();
}

void MainWindow::applyNativeTitleBarTheme() {
    if (!isVisible()) {
        return;
    }

    UiTheme::applyDarkTitleBar(this);
    QTimer::singleShot(0, this, [this]() {
        UiTheme::applyDarkTitleBar(this);
    });
    QTimer::singleShot(120, this, [this]() {
        UiTheme::applyDarkTitleBar(this);
    });
}

void MainWindow::ensureImageFilterItems() {
    if (ui->comboBox_selectImg == nullptr) {
        return;
    }

    const QStringList items = {
        QString::fromUtf8(u8"全部"),
        QString::fromUtf8(u8"已标注"),
        QString::fromUtf8(u8"未标注"),
        QString::fromUtf8(u8"含当前标签")
    };

    QSignalBlocker blocker(ui->comboBox_selectImg);
    ui->comboBox_selectImg->clear();
    for (const QString& item : items) {
        ui->comboBox_selectImg->addItem(item);
    }
    ui->comboBox_selectImg->setCurrentIndex(0);
}

void MainWindow::applyStaticTextAndIcons() {
    setWindowIcon(QIcon(QStringLiteral(":/assets/icons/app.svg")));
    ui->label_2->setText(QString::fromUtf8(u8"图片列表"));
    ui->labelTitle->setText(QString::fromUtf8(u8"标签列表"));
    ui->annTitle->setText(QString::fromUtf8(u8"标注对象"));
    ui->label_3->setText(QString::fromUtf8(u8"标注模式"));
    ui->label->setText(QString::fromUtf8(u8"当前标签"));
    ui->label_4->setText(QString::fromUtf8(u8"显示标签"));
    ui->checkBox_HideLabel->setChecked(true);
    ui->imageWidget->setShowAnnotationLabels(true);
    markSectionTitle(ui->label_2);
    markSectionTitle(ui->labelTitle);
    markSectionTitle(ui->annTitle);
    ensureImageFilterItems();

    if (ui->comboBox_AnnoMode->count() < 4) {
        ui->comboBox_AnnoMode->clear();
        ui->comboBox_AnnoMode->addItem(QString::fromUtf8(u8"自动"));
        ui->comboBox_AnnoMode->addItem(QString::fromUtf8(u8"手动"));
        ui->comboBox_AnnoMode->addItem(QString::fromUtf8(u8"小目标"));
        ui->comboBox_AnnoMode->addItem(QString::fromUtf8(u8"多目标"));
    } else {
        ui->comboBox_AnnoMode->setItemText(0, QString::fromUtf8(u8"自动"));
        ui->comboBox_AnnoMode->setItemText(1, QString::fromUtf8(u8"手动"));
        ui->comboBox_AnnoMode->setItemText(2, QString::fromUtf8(u8"小目标"));
        ui->comboBox_AnnoMode->setItemText(3, QString::fromUtf8(u8"多目标"));
    }

    configureActionButton(ui->openFolderButton, QString::fromUtf8(u8"打开文件夹"),
                          QStringLiteral(":/assets/icons/folder-open.svg"), QString::fromUtf8(u8"选择图片文件夹"));
    configureActionButton(ui->initButton, QString::fromUtf8(u8"初始化模型"),
                          QStringLiteral(":/assets/icons/cpu.svg"), QString::fromUtf8(u8"初始化 模型 推理模型"));
    configureActionButton(ui->pushButton_5, QString::fromUtf8(u8"打开目录"),
                          QStringLiteral(":/assets/icons/folder-search.svg"), QString::fromUtf8(u8"在资源管理器中打开当前目录"));
    configureActionButton(ui->pushButton_firstImg, QString::fromUtf8(u8"第一张"),
                          QStringLiteral(":/assets/icons/chevrons-left.svg"), QString::fromUtf8(u8"跳转到第一张图片"));
    configureActionButton(ui->PB_LastImg, QString::fromUtf8(u8"上一张"),
                          QStringLiteral(":/assets/icons/chevron-left.svg"), QString::fromUtf8(u8"切换到上一张图片\n快捷键A"));
    configureActionButton(ui->pushButton_InferByRects, QString::fromUtf8(u8"执行多框推理"),
                          QStringLiteral(":/assets/icons/rects-stack.svg"), QString::fromUtf8(u8"对已绘制的多个提示框执行批量推理"));
    configureActionButton(ui->pushButton_ClearRects, QString::fromUtf8(u8"清空多框"),
                          QStringLiteral(":/assets/icons/eraser.svg"), QString::fromUtf8(u8"清空当前多目标待推理框"));
    configureActionButton(ui->pushButton_nextImg, QString::fromUtf8(u8"下一张"),
                          QStringLiteral(":/assets/icons/chevron-right.svg"), QString::fromUtf8(u8"切换到下一张图片\n快捷键D"));
    configureActionButton(ui->pushButton_finalImg, QString::fromUtf8(u8"最后一张"),
                          QStringLiteral(":/assets/icons/chevrons-right.svg"), QString::fromUtf8(u8"跳转到最后一张图片"));
    configureActionButton(ui->pushButton_deleteImg, QString::fromUtf8(u8"删除图片"),
                          QStringLiteral(":/assets/icons/trash.svg"), QString::fromUtf8(u8"删除当前图片及同名 JSON"), true);
    configureActionButton(ui->addLabelButton, QString::fromUtf8(u8"添加"),
                          QStringLiteral(":/assets/icons/plus.svg"), QString::fromUtf8(u8"添加新的标注标签"));
    configureActionButton(ui->deleteLabelButton, QString::fromUtf8(u8"删除"),
                          QStringLiteral(":/assets/icons/trash.svg"), QString::fromUtf8(u8"删除选中的标签"), true);
    configureActionButton(ui->deleteAnnotationButton, QString::fromUtf8(u8"删除"),
                          QStringLiteral(":/assets/icons/trash.svg"), QString::fromUtf8(u8"删除选中的标注"), true);
    configureActionButton(ui->fixAnnotationButton, QString::fromUtf8(u8"修改标签"),
                          QStringLiteral(":/assets/icons/edit.svg"), QString::fromUtf8(u8"修改选中标注的标签"));
    configureActionButton(ui->pushButton_clearAllAnno, QString::fromUtf8(u8"清空当前图片标注"),
                          QStringLiteral(":/assets/icons/eraser.svg"), QString::fromUtf8(u8"清空当前图片的全部标注"), true);
    updateModeControls();
}

void MainWindow::onInitializeBridgeClicked() {
    startSamInitialization(false);
}

void MainWindow::startSamInitialization(bool automatic) {
    if (m_modelInitialized) {
        statusBar()->showMessage(QString::fromUtf8(u8"模型已就绪"));
        return;
    }
    if (m_modelInitializing) {
        statusBar()->showMessage(QString::fromUtf8(u8"模型正在初始化"));
        return;
    }
    if (!m_inferenceWorker) {
        const QString error = QStringLiteral("model worker thread is not available.");
        setModelStatusText(QString::fromUtf8(u8"模型初始化失败"));
        statusBar()->showMessage(QString::fromUtf8(u8"初始化失败"));
        appendLog(QStringLiteral("[Init] %1").arg(error));
        return;
    }

    m_modelInitializing = true;
    m_automaticInitialization = automatic;
    ui->initButton->setEnabled(false);
    setModelStatusText(QString::fromUtf8(u8"模型初始化中"));
    statusBar()->showMessage(QString::fromUtf8(u8"模型正在后台初始化"));

    if (automatic && m_startupOverlay) {
        m_startupOverlay->showMessage(QString::fromUtf8(u8"正在初始化模型"),
                                      QString::fromUtf8(u8"请等待初始化完成..."));
    }

    QMetaObject::invokeMethod(m_inferenceWorker, "initialize", Qt::QueuedConnection);
}

void MainWindow::requestSetCurrentImageForWorker() {
    if (!m_modelInitialized || !m_inferenceWorker || m_currentImagePath.isEmpty()) {
        return;
    }
    m_pendingWorkerImagePath = m_currentImagePath;

    QMetaObject::invokeMethod(m_inferenceWorker, "setCurrentImage", Qt::QueuedConnection,
                              Q_ARG(QString, m_currentImagePath));
}

bool MainWindow::ensureModelReadyForInference() {
    if (m_currentImagePath.isEmpty()) {
        statusBar()->showMessage(QString::fromUtf8(u8"未选择图像"));
        return false;
    }
    if (m_modelInitialized) {
        if (m_workerCurrentImagePath == m_currentImagePath) {
            return true;
        }
        requestSetCurrentImageForWorker();
        statusBar()->showMessage(QString::fromUtf8(u8"当前图像正在同步到模型，请稍后再试"));
        return false;
    }
    if (m_modelInitializing) {
        statusBar()->showMessage(QString::fromUtf8(u8"模型正在初始化，请稍后再试"));
    } else {
        statusBar()->showMessage(QString::fromUtf8(u8"推理模型未初始化"));
    }
    return false;
}

void MainWindow::syncImageStatusText() {
    if (m_currentImagePath.isEmpty()) {
        setImageStatusText(QString::fromUtf8(u8"未加载图像"));
        return;
    }
    setImageStatusText(ui->imageWidget->viewportStatusText());
}

void MainWindow::onOpenFolderClicked() {
    const QString dir = QFileDialog::getExistingDirectory(this, QString::fromUtf8(u8"打开文件夹"), m_workingDir);
    if (dir.isEmpty()) {
        return;
    }

    clearPendingMultiRects();
    setWorkingDirectory(dir);
}

void MainWindow::onImageSelectionChanged(int row) {
    if (row < 0 || row >= m_imageFilePaths.size()) {
        return;
    }

    const QString nextImagePath = m_imageFilePaths.at(row);
    if (!m_currentImagePath.isEmpty() && nextImagePath != m_currentImagePath && m_annotations.isEmpty()) {
        cleanupEmptyAnnotationFile(m_currentImagePath, QStringLiteral("SwitchImage"));
    }
    loadImageByPath(nextImagePath);
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
    if (m_annotations.isEmpty()) {
        cleanupEmptyAnnotationFile(m_currentImagePath, QStringLiteral("RemoveAnnotation"));
    }
    refreshImageList();
    statusBar()->showMessage(QString::fromUtf8(u8"标注已删除"));
}

void MainWindow::onFixAnnotationClicked() {
    if (m_currentImagePath.isEmpty()) {
        statusBar()->showMessage(QString::fromUtf8(u8"未选择图像"));
        return;
    }

    const int row = ui->annotationList->currentRow();
    if (row < 0 || row >= m_annotations.size()) {
        statusBar()->showMessage(QString::fromUtf8(u8"请选择要修改的标注"));
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
        statusBar()->showMessage(QString::fromUtf8(u8"修改标注失败"));
        appendLog(QStringLiteral("[UpdateAnnotation] %1").arg(error));
        return;
    }

    reloadAnnotationsForCurrentImage();
    refreshImageList();
    statusBar()->showMessage(QString::fromUtf8(u8"标注已修改"));
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
    if (!AnnotationJsonIO::removeAnnotationFile(m_currentImagePath, &error)) {
        statusBar()->showMessage(QString::fromUtf8(u8"清空标注失败"));
        appendLog(QStringLiteral("[ClearAllAnnotations] %1").arg(error));
        return;
    }

    m_annotations.clear();
    ui->imageWidget->setAnnotations(m_annotations);
    ui->imageWidget->clearTempResult();
    refreshAnnotationList();
    refreshImageList();
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
            ? QString::fromUtf8(u8"图像已删除，但对应 JSON 删除失败")
            : QString::fromUtf8(u8"图像已删除，当前文件夹无图像"));
        return;
    }

    statusBar()->showMessage(jsonDeleteFailed
        ? QString::fromUtf8(u8"图像已删除，但对应 JSON 删除失败")
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

void MainWindow::onInferByRectsClicked() {
    if (currentAnnotationMode() != AnnotationMode::MultiTarget) {
        statusBar()->showMessage(QString::fromUtf8(u8"请先切换到多目标模式"));
        return;
    }
    if (m_pendingMultiRects.isEmpty()) {
        statusBar()->showMessage(QString::fromUtf8(u8"请先绘制至少一个多目标提示框"));
        return;
    }
    if (!ensureModelReadyForInference()) {
        return;
    }
    if (m_inferenceBusy) {
        statusBar()->showMessage(QString::fromUtf8(u8"模型正在推理，请稍后"));
        return;
    }

    QString labelError;
    const QString labelName = currentSelectedLabel(&labelError);
    if (labelName.isEmpty()) {
        statusBar()->showMessage(QString::fromUtf8(u8"未选择标签"));
        appendLog(QStringLiteral("[InferByRects] %1").arg(labelError));
        return;
    }

    m_inferenceBusy = true;
    updateModeControls();
    statusBar()->showMessage(QString::fromUtf8(u8"模型正在执行多框推理"));
    QMetaObject::invokeMethod(m_inferenceWorker, "inferByRects", Qt::QueuedConnection,
                              Q_ARG(QVector<QRectF>, m_pendingMultiRects),
                              Q_ARG(QString, labelName),
                              Q_ARG(QString, m_currentImagePath));
}

void MainWindow::onClearRectsClicked() {
    clearPendingMultiRects();
    statusBar()->showMessage(QString::fromUtf8(u8"已清空多目标提示框"));
}

void MainWindow::onAnnotationModeChanged(int index) {
    Q_UNUSED(index);
    clearPendingMultiRects();
    updateModeControls();
}

void MainWindow::onImageFilterChanged(int index) {
    Q_UNUSED(index);
    refreshImageList();
}

void MainWindow::onCurrentLabelFilterChanged(int index) {
    Q_UNUSED(index);
    if (currentImageFilterMode() == ImageFilterMode::ContainsCurrentLabel) {
        refreshImageList();
    }
}

void MainWindow::onPointPromptRequested(const QPointF& imagePoint) {
    const AnnotationMode mode = currentAnnotationMode();
    if (mode != AnnotationMode::Auto) {
        Q_UNUSED(imagePoint);
        if (mode == AnnotationMode::SmallTarget) {
            statusBar()->showMessage(QString::fromUtf8(u8"小目标模式请拖拽矩形框进行局部推理"));
        } else if (mode == AnnotationMode::MultiTarget) {
            statusBar()->showMessage(QString::fromUtf8(u8"多目标模式请拖拽多个矩形框"));
        } else {
            statusBar()->showMessage(QString::fromUtf8(u8"手动模式请拖拽矩形标注"));
        }
        return;
    }

    if (!ensureModelReadyForInference()) {
        return;
    }
    if (m_inferenceBusy) {
        statusBar()->showMessage(QString::fromUtf8(u8"模型正在推理，请稍后"));
        return;
    }

    QString labelError;
    const QString labelName = currentSelectedLabel(&labelError);
    if (labelName.isEmpty()) {
        statusBar()->showMessage(QString::fromUtf8(u8"未选择标签"));
        appendLog(QStringLiteral("[InferByPoint] %1").arg(labelError));
        return;
    }

    m_inferenceBusy = true;
    updateModeControls();
    statusBar()->showMessage(QString::fromUtf8(u8"模型正在按点推理"));
    QMetaObject::invokeMethod(m_inferenceWorker, "inferByPoint", Qt::QueuedConnection,
                              Q_ARG(QPointF, imagePoint),
                              Q_ARG(QString, labelName),
                              Q_ARG(QString, m_currentImagePath));
}

void MainWindow::onRectPromptRequested(const QRectF& imageRect) {
    const AnnotationMode mode = currentAnnotationMode();
    if (mode == AnnotationMode::Manual) {
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

    if (mode == AnnotationMode::MultiTarget) {
        const QRectF rect = imageRect.normalized();
        if (!rect.isValid() || rect.width() < 2.0 || rect.height() < 2.0) {
            statusBar()->showMessage(QString::fromUtf8(u8"多目标提示框太小"));
            appendLog(QStringLiteral("[InferByRects] prompt rectangle is too small"));
            return;
        }
        m_pendingMultiRects.push_back(rect);
        ui->imageWidget->setPendingPromptRects(m_pendingMultiRects);
        updateModeControls();
        statusBar()->showMessage(QString::fromUtf8(u8"已添加第 %1 个多目标提示框").arg(m_pendingMultiRects.size()));
        return;
    }

    if (!ensureModelReadyForInference()) {
        return;
    }
    if (m_inferenceBusy) {
        statusBar()->showMessage(QString::fromUtf8(u8"模型正在推理，请稍后"));
        return;
    }

    QString labelError;
    const QString labelName = currentSelectedLabel(&labelError);
    if (labelName.isEmpty()) {
        statusBar()->showMessage(QString::fromUtf8(u8"未选择标签"));
        appendLog(QStringLiteral("[InferByRect] %1").arg(labelError));
        return;
    }

    m_inferenceBusy = true;
    updateModeControls();
    if (mode == AnnotationMode::SmallTarget) {
        statusBar()->showMessage(QString::fromUtf8(u8"模型正在执行小目标局部推理"));
        QMetaObject::invokeMethod(m_inferenceWorker, "inferSmallTargetByRect", Qt::QueuedConnection,
                                  Q_ARG(QRectF, imageRect),
                                  Q_ARG(QString, labelName),
                                  Q_ARG(QString, m_currentImagePath));
    } else {
        statusBar()->showMessage(QString::fromUtf8(u8"模型正在按框推理"));
        QMetaObject::invokeMethod(m_inferenceWorker, "inferByRect", Qt::QueuedConnection,
                                  Q_ARG(QRectF, imageRect),
                                  Q_ARG(QString, labelName),
                                  Q_ARG(QString, m_currentImagePath));
    }
}

void MainWindow::onAnnotationSelectionChanged(int annotationIndex) {
    ui->imageWidget->setSelectedAnnotationIndex(annotationIndex);
    if (annotationIndex >= 0 && annotationIndex < ui->annotationList->count()) {
        ui->annotationList->setCurrentRow(annotationIndex);
    }
}

void MainWindow::onImageViewportChanged(const QString& statusText) {
    setImageStatusText(statusText);
}

void MainWindow::onSamInitializeFinished(bool success, const QString& errorMessage) {
    m_modelInitializing = false;
    ui->initButton->setEnabled(true);
    if (m_startupOverlay && m_automaticInitialization) {
        m_startupOverlay->hideOverlay();
    }
    m_automaticInitialization = false;
    applyNativeTitleBarTheme();

    if (!success) {
        m_modelInitialized = false;
        setModelStatusText(QString::fromUtf8(u8"模型初始化失败"));
        statusBar()->showMessage(QString::fromUtf8(u8"模型初始化失败"));
        appendLog(QStringLiteral("[Init] %1").arg(errorMessage));
        return;
    }

    m_modelInitialized = true;
    setModelStatusText(QString::fromUtf8(u8"模型已就绪"));
    statusBar()->showMessage(QString::fromUtf8(u8"模型初始化成功"));
    appendLog(QStringLiteral("[Init] model initialized"));
    requestSetCurrentImageForWorker();
}

void MainWindow::onSamCurrentImageFinished(const QString& imagePath, bool success, const QString& errorMessage) {
    if (imagePath != m_currentImagePath) {
        return;
    }
    if (!success) {
        if (imagePath == m_pendingWorkerImagePath) {
            m_pendingWorkerImagePath.clear();
        }
        statusBar()->showMessage(QString::fromUtf8(u8"设置图像失败"));
        appendLog(QStringLiteral("[SetCurrentImage] %1").arg(errorMessage));
        return;
    }
    m_workerCurrentImagePath = imagePath;
    m_pendingWorkerImagePath.clear();
    statusBar()->showMessage(QString::fromUtf8(u8"图像已同步到模型"));
}

void MainWindow::onSamPointInferenceFinished(const SamInferResult& result, const QString& labelName,
                                             const QString& imagePath) {
    m_inferenceBusy = false;
    updateModeControls();
    if (!result.success) {
        statusBar()->showMessage(QString::fromUtf8(u8"推理失败"));
        appendLog(QStringLiteral("[InferByPoint] %1").arg(result.errorMessage));
        return;
    }
    saveSamResultAnnotations(result, labelName, imagePath);
}

void MainWindow::onSamRectInferenceFinished(const SamInferResult& result, const QString& labelName,
                                            const QString& imagePath) {
    m_inferenceBusy = false;
    updateModeControls();
    if (!result.success) {
        statusBar()->showMessage(QString::fromUtf8(u8"推理失败"));
        appendLog(QStringLiteral("[InferByRect] %1").arg(result.errorMessage));
        return;
    }
    saveSamResultAnnotations(result, labelName, imagePath);
}

void MainWindow::onSamRectsInferenceFinished(const SamInferResult& result, const QString& labelName,
                                             const QString& imagePath) {
    m_inferenceBusy = false;
    updateModeControls();
    if (!result.success) {
        statusBar()->showMessage(QString::fromUtf8(u8"多框推理失败"));
        appendLog(QStringLiteral("[InferByRects] %1").arg(result.errorMessage));
        return;
    }
    if (saveSamResultAnnotations(result, labelName, imagePath) && imagePath == m_currentImagePath) {
        clearPendingMultiRects();
    }
}

void MainWindow::setupConnections() {
    QShortcut *lastImageShortcut = new QShortcut(QKeySequence(Qt::Key_A), this);
    lastImageShortcut->setContext(Qt::WidgetWithChildrenShortcut);
    connect(lastImageShortcut, &QShortcut::activated, this, &MainWindow::onPreviousImageClicked);

    QShortcut *nextImageShortcut = new QShortcut(QKeySequence(Qt::Key_D), this);
    nextImageShortcut->setContext(Qt::WidgetWithChildrenShortcut);
    connect(nextImageShortcut, &QShortcut::activated, this, &MainWindow::onNextImageClicked);

    connect(ui->initButton, &QPushButton::clicked, this, &MainWindow::onInitializeBridgeClicked);
    connect(ui->openFolderButton, &QPushButton::clicked, this, &MainWindow::onOpenFolderClicked);
    connect(ui->pushButton_5, &QPushButton::clicked, this, &MainWindow::onOpenCurrentFolderClicked);
    connect(ui->pushButton_clearAllAnno, &QPushButton::clicked, this, &MainWindow::onClearAllAnnotationsClicked);
    connect(ui->pushButton_firstImg, &QPushButton::clicked, this, &MainWindow::onFirstImageClicked);
    connect(ui->PB_LastImg, &QPushButton::clicked, this, &MainWindow::onPreviousImageClicked);
    connect(ui->pushButton_InferByRects, &QPushButton::clicked, this, &MainWindow::onInferByRectsClicked);
    connect(ui->pushButton_ClearRects, &QPushButton::clicked, this, &MainWindow::onClearRectsClicked);
    connect(ui->pushButton_deleteImg, &QPushButton::clicked, this, &MainWindow::onDeleteCurrentImageClicked);
    connect(ui->pushButton_nextImg, &QPushButton::clicked, this, &MainWindow::onNextImageClicked);
    connect(ui->pushButton_finalImg, &QPushButton::clicked, this, &MainWindow::onFinalImageClicked);

    connect(ui->addLabelButton, &QPushButton::clicked, this, &MainWindow::onAddLabelClicked);
    connect(ui->deleteLabelButton, &QPushButton::clicked, this, &MainWindow::onDeleteLabelClicked);
    connect(ui->deleteAnnotationButton, &QPushButton::clicked, this, &MainWindow::onDeleteAnnotationClicked);
    connect(ui->fixAnnotationButton, &QPushButton::clicked, this, &MainWindow::onFixAnnotationClicked);

    connect(ui->imageList, &QListWidget::currentRowChanged, this, &MainWindow::onImageSelectionChanged);
    connect(ui->comboBox_AnnoMode, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MainWindow::onAnnotationModeChanged);
    connect(ui->comboBox_selectImg, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MainWindow::onImageFilterChanged);
    connect(ui->comboBox_CurrentLabel, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MainWindow::onCurrentLabelFilterChanged);
    connect(ui->checkBox_HideLabel, &QCheckBox::toggled, ui->imageWidget,
            &ImageAnnotateWidget::setShowAnnotationLabels);

    connect(ui->imageWidget, &ImageAnnotateWidget::pointPromptRequested, this, &MainWindow::onPointPromptRequested);
    connect(ui->imageWidget, &ImageAnnotateWidget::rectPromptRequested, this, &MainWindow::onRectPromptRequested);
    connect(ui->imageWidget, &ImageAnnotateWidget::annotationSelectionChanged,
            this, &MainWindow::onAnnotationSelectionChanged);
    connect(ui->imageWidget, &ImageAnnotateWidget::viewportStatusChanged,
            this, &MainWindow::onImageViewportChanged);

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
        setWindowTitle(QString::fromUtf8(u8"颖图半自动标注软件 [%1]").arg(QDir::toNativeSeparators(m_workingDir)));
    }
    applyNativeTitleBarTheme();
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
    updateStatusSummary();
}

void MainWindow::refreshImageList() {
    const QString previousCurrentImagePath = m_currentImagePath;
    QSignalBlocker blocker(ui->imageList);

    ui->imageList->clear();
    m_allImageFilePaths.clear();
    m_imageFilePaths.clear();

    if (m_workingDir.isEmpty()) {
        updateStatusSummary();
        return;
    }

    QDir dir(m_workingDir);
    const QStringList filters = {QStringLiteral("*.jpg"), QStringLiteral("*.jpeg"), QStringLiteral("*.png"), QStringLiteral("*.bmp")};
    const QFileInfoList files = dir.entryInfoList(filters, QDir::Files | QDir::Readable | QDir::NoSymLinks, QDir::Name);

    const ImageFilterMode filterMode = currentImageFilterMode();
    const QString currentLabel = ui->comboBox_CurrentLabel->currentText();
    const bool missingLabelForFilter = filterMode == ImageFilterMode::ContainsCurrentLabel && currentLabel.isEmpty();
    int rowToKeep = -1;

    for (const QFileInfo& fi : files) {
        const QString imagePath = fi.absoluteFilePath();
        m_allImageFilePaths.push_back(imagePath);

        bool hasAnnotations = false;
        QString error;
        if (!AnnotationJsonIO::hasValidAnnotations(imagePath, &hasAnnotations, &error)) {
            appendLog(QStringLiteral("[ImageList] failed to inspect annotations for %1: %2").arg(imagePath, error));
            hasAnnotations = false;
        }

        bool visible = false;
        switch (filterMode) {
        case ImageFilterMode::Annotated:
            visible = hasAnnotations;
            break;
        case ImageFilterMode::Unannotated:
            visible = !hasAnnotations;
            break;
        case ImageFilterMode::ContainsCurrentLabel: {
            if (!missingLabelForFilter) {
                bool containsLabel = false;
                if (!AnnotationJsonIO::annotationFileContainsLabel(imagePath, currentLabel, &containsLabel, &error)) {
                    appendLog(QStringLiteral("[ImageList] failed to inspect label for %1: %2").arg(imagePath, error));
                    containsLabel = false;
                }
                visible = containsLabel;
            }
            break;
        }
        case ImageFilterMode::All:
        default:
            visible = true;
            break;
        }

        if (!visible) {
            continue;
        }

        if (imagePath == previousCurrentImagePath) {
            rowToKeep = m_imageFilePaths.size();
        }

        m_imageFilePaths.push_back(imagePath);
        auto* item = new QListWidgetItem(QIcon(hasAnnotations
                                                   ? QStringLiteral(":/assets/icons/status-checked.svg")
                                                   : QStringLiteral(":/assets/icons/status-unchecked.svg")),
                                         fi.fileName());
        ui->imageList->addItem(item);
    }

    if (rowToKeep >= 0) {
        ui->imageList->setCurrentRow(rowToKeep);
    }

    blocker.unblock();

    if (m_imageFilePaths.isEmpty()) {
        clearCurrentImageState();
        if (missingLabelForFilter) {
            statusBar()->showMessage(QString::fromUtf8(u8"请先选择当前标签"));
        } else if (!m_allImageFilePaths.isEmpty()) {
            statusBar()->showMessage(QString::fromUtf8(u8"当前筛选条件下没有图片"));
        }
        return;
    }

    if (rowToKeep < 0) {
        ui->imageList->setCurrentRow(0);
    }

    updateStatusSummary();
}

bool MainWindow::cleanupEmptyAnnotationFile(const QString& imagePath, const QString& reason) {
    if (imagePath.isEmpty()) {
        return true;
    }

    const QString jsonPath = AnnotationJsonIO::annotationFilePath(imagePath);
    if (!QFileInfo::exists(jsonPath)) {
        return true;
    }

    QList<AnnotationObject> annotations;
    QString error;
    if (!AnnotationJsonIO::loadAnnotations(imagePath, &annotations, &error)) {
        appendLog(QStringLiteral("[%1] keep unreadable annotation file %2: %3").arg(reason, jsonPath, error));
        return false;
    }

    if (!annotations.isEmpty()) {
        return true;
    }

    if (!AnnotationJsonIO::removeAnnotationFile(imagePath, &error)) {
        appendLog(QStringLiteral("[%1] failed to delete empty annotation file %2: %3").arg(reason, jsonPath, error));
        return false;
    }

    appendLog(QStringLiteral("[%1] deleted empty annotation file: %2").arg(reason, jsonPath));
    return true;
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
    clearPendingMultiRects();
    ui->imageWidget->clearTempResult();

    if (!reloadAnnotationsForCurrentImage()) {
        return false;
    }
    if (m_annotations.isEmpty()) {
        cleanupEmptyAnnotationFile(m_currentImagePath, QStringLiteral("LoadImage"));
    }
    syncImageStatusText();

    if (m_modelInitialized) {
        requestSetCurrentImageForWorker();
        statusBar()->showMessage(QString::fromUtf8(u8"图像已加载，正在同步到模型"));
    } else if (m_modelInitializing) {
        statusBar()->showMessage(QString::fromUtf8(u8"图像已加载，模型正在初始化"));
    } else {
        statusBar()->showMessage(QString::fromUtf8(u8"图像已加载，推理模型未初始化"));
    }

    updateStatusSummary();
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
        syncImageStatusText();
        updateStatusSummary();
        return true;
    }

    QString error;
    if (!AnnotationJsonIO::loadAnnotations(m_currentImagePath, &m_annotations, &error)) {
        statusBar()->showMessage(QString::fromUtf8(u8"加载标注失败"));
        appendLog(QStringLiteral("[LoadAnnotations] %1").arg(error));
        return false;
    }

    if (!normalizeAnnotationsForCurrentImage()) {
        return false;
    }

    updateAnnotationColors();
    ui->imageWidget->setAnnotations(m_annotations);
    refreshAnnotationList();
    syncImageStatusText();
    updateStatusSummary();
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

QList<AnnotationObject> MainWindow::nmsAnnotations(const QList<AnnotationObject>& annotations) const {
    QList<AnnotationObject> kept;
    kept.reserve(annotations.size());

    for (const AnnotationObject& candidate : annotations) {
        const QRectF candidateRect = rectFromAnnotation(candidate);
        if (!candidateRect.isValid() || candidateRect.width() <= 0.0 || candidateRect.height() <= 0.0) {
            continue;
        }

        bool suppressed = false;
        for (const AnnotationObject& existing : kept) {
            if (existing.label != candidate.label) {
                continue;
            }
            if (rectIou(candidateRect, rectFromAnnotation(existing)) > kNmsIouThreshold) {
                suppressed = true;
                break;
            }
        }

        if (!suppressed) {
            AnnotationObject normalized = candidate;
            normalized.shapeIndex = kept.size();
            normalized.rectPolygonImage = toAxisAlignedRectPolygon(candidate.rectPolygonImage);
            kept.push_back(normalized);
        }
    }

    return kept;
}

bool MainWindow::normalizeAnnotationsForCurrentImage() {
    if (m_currentImagePath.isEmpty()) {
        return true;
    }

    const int originalCount = m_annotations.size();
    const QList<AnnotationObject> filtered = nmsAnnotations(m_annotations);
    if (filtered.size() == originalCount) {
        m_annotations = filtered;
        return true;
    }

    QString error;
    if (!AnnotationJsonIO::replaceAnnotations(m_currentImagePath, filtered, &error)) {
        statusBar()->showMessage(QString::fromUtf8(u8"标注去重保存失败"));
        appendLog(QStringLiteral("[AnnotationNMS] %1").arg(error));
        return false;
    }

    m_annotations = filtered;
    appendLog(QStringLiteral("[AnnotationNMS] removed %1 duplicate annotations")
                  .arg(originalCount - filtered.size()));
    return true;
}

bool MainWindow::normalizeAnnotationsForImage(const QString& imagePath) {
    if (imagePath.isEmpty()) {
        return true;
    }
    if (imagePath == m_currentImagePath) {
        return normalizeAnnotationsForCurrentImage();
    }

    QList<AnnotationObject> annotations;
    QString error;
    if (!AnnotationJsonIO::loadAnnotations(imagePath, &annotations, &error)) {
        appendLog(QStringLiteral("[AnnotationNMS] failed to load %1: %2").arg(imagePath, error));
        return false;
    }

    const int originalCount = annotations.size();
    const QList<AnnotationObject> filtered = nmsAnnotations(annotations);
    if (filtered.size() == originalCount) {
        return true;
    }
    if (!AnnotationJsonIO::replaceAnnotations(imagePath, filtered, &error)) {
        appendLog(QStringLiteral("[AnnotationNMS] failed to save %1: %2").arg(imagePath, error));
        return false;
    }
    appendLog(QStringLiteral("[AnnotationNMS] removed %1 duplicate annotations from %2")
                  .arg(originalCount - filtered.size())
                  .arg(imagePath));
    return true;
}

void MainWindow::updateStatusSummary(const QString& message) {
    const QString folder = m_workingDir.isEmpty()
        ? QString::fromUtf8(u8"未设置目录")
        : QDir::toNativeSeparators(m_workingDir);
    const QString current = m_currentImagePath.isEmpty()
        ? QString::fromUtf8(u8"未选择")
        : QFileInfo(m_currentImagePath).fileName();
    if (m_folderStatusLabel) {
        m_folderStatusLabel->setText(QString::fromUtf8(u8"目录: %1 | 图片: %2 | 当前: %3")
                                         .arg(folder)
                                         .arg(m_imageFilePaths.size())
                                         .arg(current));
    }
    if (!message.isEmpty()) {
        statusBar()->showMessage(message);
    }
}

void MainWindow::setModelStatusText(const QString& text) {
    if (m_modelStatusLabel) {
        m_modelStatusLabel->setText(text);
    }
}

void MainWindow::setImageStatusText(const QString& text) {
    if (m_imageStatusLabel) {
        m_imageStatusLabel->setText(text);
    }
}

int MainWindow::colorForLabel(const QString& label) const {
    const int idx = m_labelConfig.nameList.indexOf(label);
    if (idx >= 0 && idx < m_labelConfig.colorDefine.size()) {
        return m_labelConfig.colorDefine.at(idx);
    }
    return 0x00C8FF;
}

MainWindow::AnnotationMode MainWindow::currentAnnotationMode() const {
    switch (ui->comboBox_AnnoMode->currentIndex()) {
    case 1:
        return AnnotationMode::Manual;
    case 2:
        return AnnotationMode::SmallTarget;
    case 3:
        return AnnotationMode::MultiTarget;
    case 0:
    default:
        return AnnotationMode::Auto;
    }
}

MainWindow::ImageFilterMode MainWindow::currentImageFilterMode() const {
    switch (ui->comboBox_selectImg->currentIndex()) {
    case 1:
        return ImageFilterMode::Annotated;
    case 2:
        return ImageFilterMode::Unannotated;
    case 3:
        return ImageFilterMode::ContainsCurrentLabel;
    case 0:
    default:
        return ImageFilterMode::All;
    }
}

bool MainWindow::isAutoAnnotationMode() const {
    return currentAnnotationMode() == AnnotationMode::Auto;
}

void MainWindow::updateModeControls() {
    const bool multiMode = currentAnnotationMode() == AnnotationMode::MultiTarget;
    ui->pushButton_InferByRects->setEnabled(multiMode && !m_pendingMultiRects.isEmpty() && !m_inferenceBusy);
    ui->pushButton_ClearRects->setEnabled(multiMode && !m_pendingMultiRects.isEmpty() && !m_inferenceBusy);
}

void MainWindow::clearPendingMultiRects() {
    if (m_pendingMultiRects.isEmpty()) {
        ui->imageWidget->clearPendingPromptRects();
        updateModeControls();
        return;
    }
    m_pendingMultiRects.clear();
    ui->imageWidget->clearPendingPromptRects();
    updateModeControls();
}

void MainWindow::clearCurrentImageState() {
    m_currentImagePath.clear();
    m_annotations.clear();
    clearPendingMultiRects();
    ui->imageWidget->setAnnotations(m_annotations);
    ui->imageWidget->clearTempResult();
    ui->imageWidget->setVisible(true);
    ui->annotationList->clear();
    setImageStatusText(QString::fromUtf8(u8"未加载图像"));
    updateStatusSummary();
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
    refreshImageList();
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
            *errorMessage = result.errorMessage.isEmpty() ? QStringLiteral("model inference failed.") : result.errorMessage;
        }
        return annotations;
    }

    for (const SamObjectResult& object : result.objects) {
        if (!object.success) {
            appendLog(QStringLiteral("[AI Result] skip unsuccessful object: %1").arg(object.errorMessage));
            continue;
        }

        QPolygonF rect = polygonFromRoiData(object.minRectRoiData);
        if (rect.size() != 4) {
            rect = polygonFromRoiData(object.roiData);
        }
        if (rect.size() != 4) {
            appendLog(QStringLiteral("[AI Result] skip object with invalid geometry, score=%1").arg(object.score));
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

bool MainWindow::saveSamResultAnnotations(const SamInferResult& result, const QString& labelName,
                                          const QString& imagePath) {
    const QString targetImagePath = imagePath.isEmpty() ? m_currentImagePath : imagePath;
    if (targetImagePath.isEmpty()) {
        statusBar()->showMessage(QString::fromUtf8(u8"未选择图像"));
        appendLog(QStringLiteral("[AI SaveResult] target image path is empty."));
        return false;
    }

    QString error;
    const QList<AnnotationObject> newAnnotations = annotationsFromSamResult(result, labelName, &error);
    if (newAnnotations.isEmpty()) {
        statusBar()->showMessage(QString::fromUtf8(u8"推理结果无有效目标"));
        appendLog(QStringLiteral("[AI SaveResult] %1").arg(error));
        return false;
    }

    if (!AnnotationJsonIO::appendAnnotations(targetImagePath, newAnnotations, &error)) {
        statusBar()->showMessage(QString::fromUtf8(u8"保存AI标注失败"));
        appendLog(QStringLiteral("[AI SaveResult] %1").arg(error));
        return false;
    }

    if (targetImagePath == m_currentImagePath) {
        ui->imageWidget->clearTempResult();
        reloadAnnotationsForCurrentImage();
    } else if (!normalizeAnnotationsForImage(targetImagePath)) {
        statusBar()->showMessage(QString::fromUtf8(u8"保存AI标注后去重失败"));
        return false;
    }
    statusBar()->showMessage(QString::fromUtf8(u8"AI已保存 %1 个目标").arg(newAnnotations.size()));
    refreshImageList();
    appendLog(QStringLiteral("[AI SaveResult] appended %1 annotations with label %2")
                  .arg(newAnnotations.size())
                  .arg(labelName));
    return true;
}
