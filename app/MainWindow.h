#ifndef AUTOLABELPROJECT_APP_MAINWINDOW_H
#define AUTOLABELPROJECT_APP_MAINWINDOW_H

#include <QMainWindow>
#include <QList>
#include <QStringList>

#include "app/AppTypes.h"
#include "inference/SamTypes.h"

class QListWidget;
class QPushButton;

class SamInferenceBridge;
class ImageAnnotateWidget;

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

    void onPointPromptRequested(const QPointF& imagePoint);
    void onRectPromptRequested(const QRectF& imageRect);
    void onAnnotationSelectionChanged(int annotationIndex);

private:
    void setupUi();
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
    QString currentSelectedLabel(QString* errorMessage = nullptr) const;
    QList<AnnotationObject> annotationsFromSamResult(const SamInferResult& result, const QString& labelName,
                                                     QString* errorMessage = nullptr);
    bool saveSamResultAnnotations(const SamInferResult& result, const QString& labelName);

    QListWidget* m_imageList = nullptr;
    ImageAnnotateWidget* m_imageWidget = nullptr;
    QListWidget* m_labelList = nullptr;
    QListWidget* m_annotationList = nullptr;

    QPushButton* m_initButton = nullptr;
    QPushButton* m_openFolderButton = nullptr;

    QPushButton* m_addLabelButton = nullptr;
    QPushButton* m_deleteLabelButton = nullptr;

    QPushButton* m_deleteAnnotationButton = nullptr;
    QPushButton* m_fixAnnotationButton = nullptr;

    SamInferenceBridge* m_bridge = nullptr;

    QString m_workingDir;
    QStringList m_imageFilePaths;
    QString m_currentImagePath;
    QString m_labelConfigPath;

    QList<AnnotationObject> m_annotations;
    LabelConfig m_labelConfig;
};

#endif // AUTOLABELPROJECT_APP_MAINWINDOW_H
