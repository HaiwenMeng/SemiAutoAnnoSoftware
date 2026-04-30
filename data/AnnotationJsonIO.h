#ifndef AUTOLABELPROJECT_DATA_ANNOTATIONJSONIO_H
#define AUTOLABELPROJECT_DATA_ANNOTATIONJSONIO_H

#include <QList>
#include <QString>

#include "app/AppTypes.h"

class AnnotationJsonIO {
public:
    static QString annotationFilePath(const QString& imagePath);
    static bool hasValidAnnotations(const QString& imagePath, bool* hasAnnotations, QString* errorMessage = nullptr);
    static bool removeAnnotationFile(const QString& imagePath, QString* errorMessage = nullptr);
    static bool annotationFileContainsLabel(const QString& imagePath, const QString& labelName, bool* containsLabel,
                                            QString* errorMessage = nullptr);
    static bool loadAnnotations(const QString& imagePath, QList<AnnotationObject>* annotations, QString* errorMessage = nullptr);
    static bool appendAnnotation(const QString& imagePath, const AnnotationObject& annotation, QString* errorMessage = nullptr);
    static bool appendAnnotations(const QString& imagePath, const QList<AnnotationObject>& annotations,
                                  QString* errorMessage = nullptr);
    static bool replaceAnnotations(const QString& imagePath, const QList<AnnotationObject>& annotations,
                                   QString* errorMessage = nullptr);
    static bool clearAnnotations(const QString& imagePath, QString* errorMessage = nullptr);
    static bool removeAnnotationByIndex(const QString& imagePath, int shapeIndex, QString* errorMessage = nullptr);
    static bool updateAnnotationByIndex(const QString& imagePath, int shapeIndex, const AnnotationObject& annotation,
                                        QString* errorMessage = nullptr);

private:
    static QString jsonPathFromImagePath(const QString& imagePath);
};

#endif // AUTOLABELPROJECT_DATA_ANNOTATIONJSONIO_H
