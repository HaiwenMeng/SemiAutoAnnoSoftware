#ifndef AUTOLABELPROJECT_APP_MAINWINDOW_H
#define AUTOLABELPROJECT_APP_MAINWINDOW_H

#include <QMainWindow>
#include <QList>
#include <QStringList>

#include "app/AppTypes.h"
#include "inference/SamTypes.h"

class SamInferenceBridge;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override;

private slots:
    void onInitializeBridgeClicked();
    void onOpenFolderClicked();
    void onImageSelectionChanged(int row);

    void onDeleteAnnotationClicked();
    void onFixAnnotationClicked();
    void onAddLabelClicked();
    void onDeleteLabelClicked();
    void onOpenCurrentFolderClicked();
    void onClearAllAnnotationsClicked();
    void onFirstImageClicked();
    void onPreviousImageClicked();
    void onDeleteCurrentImageClicked();
    void onNextImageClicked();
    void onFinalImageClicked();

    void onPointPromptRequested(const QPointF& imagePoint);
    void onRectPromptRequested(const QRectF& imageRect);
    void onAnnotationSelectionChanged(int annotationIndex);

private:
    void setupConnections();
    void appendLog(const QString& message);

    void updateWindowTitle();
    void setWorkingDirectory(const QString& folderPath);
    void refreshImageList();
    bool loadImageByPath(const QString& imagePath);

    bool loadLabelConfig();
    bool saveLabelConfig();
    void refreshLabelList();
    bool reloadAnnotationsForCurrentImage();
    void refreshAnnotationList();
    void updateAnnotationColors();
    int colorForLabel(const QString& label) const;
    bool isAutoAnnotationMode() const;
    void clearCurrentImageState();
    bool saveManualRectAnnotation(const QRectF& imageRect, const QString& labelName);
    QString currentSelectedLabel(QString* errorMessage = nullptr) const;
    QList<AnnotationObject> annotationsFromSamResult(const SamInferResult& result, const QString& labelName,
                                                     QString* errorMessage = nullptr);
    bool saveSamResultAnnotations(const SamInferResult& result, const QString& labelName);

    Ui::MainWindow* ui = nullptr;
    SamInferenceBridge* m_bridge = nullptr;

    QString m_workingDir;
    QStringList m_imageFilePaths;
    QString m_currentImagePath;
    QString m_labelConfigPath;

    QList<AnnotationObject> m_annotations;
    LabelConfig m_labelConfig;
};

#endif // AUTOLABELPROJECT_APP_MAINWINDOW_H
