#include "data/AnnotationJsonIO.h"

#include <QFile>
#include <QFileInfo>
#include <QImageReader>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

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

QPolygonF polygonFromShapePoints(const QJsonArray& pointsArray) {
    QPolygonF poly;
    for (const QJsonValue& pointValue : pointsArray) {
        if (!pointValue.isArray()) {
            continue;
        }
        const QJsonArray p = pointValue.toArray();
        if (p.size() < 2) {
            continue;
        }
        poly << QPointF(p.at(0).toDouble(), p.at(1).toDouble());
    }
    return toAxisAlignedRectPolygon(poly);
}

QJsonArray pointsArrayFromRectPolygon(const QPolygonF& rectPoly) {
    QJsonArray points;
    const QPolygonF rect = toAxisAlignedRectPolygon(rectPoly);
    for (const QPointF& p : rect) {
        QJsonArray point;
        point.append(p.x());
        point.append(p.y());
        points.append(point);
    }
    return points;
}

QJsonObject createDefaultRoot(const QString& imagePath) {
    QFileInfo info(imagePath);
    QImageReader reader(imagePath);
    const QSize size = reader.size();

    QJsonObject root;
    root.insert(QStringLiteral("imageData"), QJsonValue::Null);
    root.insert(QStringLiteral("imageHeight"), size.height() > 0 ? size.height() : 0);
    root.insert(QStringLiteral("imageWidth"), size.width() > 0 ? size.width() : 0);
    root.insert(QStringLiteral("imagePath"), info.fileName());
    root.insert(QStringLiteral("shapes"), QJsonArray());
    return root;
}

void refreshImageFields(QJsonObject* root, const QString& imagePath) {
    if (root == nullptr) {
        return;
    }
    QFileInfo info(imagePath);
    QImageReader reader(imagePath);
    const QSize size = reader.size();
    root->insert(QStringLiteral("imageData"), QJsonValue::Null);
    root->insert(QStringLiteral("imageHeight"), size.height() > 0 ? size.height() : 0);
    root->insert(QStringLiteral("imageWidth"), size.width() > 0 ? size.width() : 0);
    root->insert(QStringLiteral("imagePath"), info.fileName());
}

QJsonObject shapeObjectFromAnnotation(const AnnotationObject& annotation) {
    QJsonObject obj;
    obj.insert(QStringLiteral("label"), annotation.label);
    obj.insert(QStringLiteral("shape_type"), QStringLiteral("rectangle"));
    obj.insert(QStringLiteral("points"), pointsArrayFromRectPolygon(annotation.rectPolygonImage));
    return obj;
}

bool loadRootObject(const QString& jsonPath, const QString& imagePath, QJsonObject* root, QString* errorMessage) {
    if (root == nullptr) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("Internal error: null root");
        }
        return false;
    }

    QFile file(jsonPath);
    if (!file.exists()) {
        *root = createDefaultRoot(imagePath);
        return true;
    }

    if (!file.open(QIODevice::ReadOnly)) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("Failed to open json: %1").arg(jsonPath);
        }
        return false;
    }

    const QByteArray data = file.readAll();
    file.close();

    if (data.trimmed().isEmpty()) {
        *root = createDefaultRoot(imagePath);
        return true;
    }

    QJsonParseError parseError{};
    const QJsonDocument doc = QJsonDocument::fromJson(data, &parseError);
    if (parseError.error != QJsonParseError::NoError || !doc.isObject()) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("Invalid json format: %1").arg(jsonPath);
        }
        return false;
    }

    *root = doc.object();
    if (!root->contains(QStringLiteral("shapes")) || !(*root)[QStringLiteral("shapes")].isArray()) {
        root->insert(QStringLiteral("shapes"), QJsonArray());
    }
    if (!root->contains(QStringLiteral("imageData"))) {
        root->insert(QStringLiteral("imageData"), QJsonValue::Null);
    }

    return true;
}

bool saveRootObject(const QString& jsonPath, const QJsonObject& root, QString* errorMessage) {
    QFile file(jsonPath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("Failed to write json: %1").arg(jsonPath);
        }
        return false;
    }

    const QJsonDocument doc(root);
    file.write(doc.toJson(QJsonDocument::Indented));
    file.close();
    return true;
}
}

QString AnnotationJsonIO::jsonPathFromImagePath(const QString& imagePath) {
    QFileInfo info(imagePath);
    return info.absolutePath() + QLatin1Char('/') + info.completeBaseName() + QStringLiteral(".json");
}

bool AnnotationJsonIO::loadAnnotations(const QString& imagePath, QList<AnnotationObject>* annotations, QString* errorMessage) {
    if (annotations == nullptr) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("Internal error: null annotations");
        }
        return false;
    }

    annotations->clear();
    const QString jsonPath = jsonPathFromImagePath(imagePath);

    QJsonObject root;
    if (!loadRootObject(jsonPath, imagePath, &root, errorMessage)) {
        return false;
    }

    const QJsonArray shapes = root.value(QStringLiteral("shapes")).toArray();
    for (int i = 0; i < shapes.size(); ++i) {
        if (!shapes.at(i).isObject()) {
            continue;
        }
        const QJsonObject obj = shapes.at(i).toObject();
        const QString shapeType = obj.value(QStringLiteral("shape_type")).toString();
        if (!shapeType.isEmpty() && shapeType != QStringLiteral("rectangle")) {
            continue;
        }

        const QJsonArray pointsArray = obj.value(QStringLiteral("points")).toArray();
        const QPolygonF rectPoly = polygonFromShapePoints(pointsArray);
        if (rectPoly.size() != 4) {
            continue;
        }

        AnnotationObject ann;
        ann.shapeIndex = i;
        ann.label = obj.value(QStringLiteral("label")).toString();
        ann.rectPolygonImage = rectPoly;
        annotations->push_back(ann);
    }

    return true;
}

bool AnnotationJsonIO::appendAnnotation(const QString& imagePath, const AnnotationObject& annotation, QString* errorMessage) {
    return appendAnnotations(imagePath, QList<AnnotationObject>{annotation}, errorMessage);
}

bool AnnotationJsonIO::appendAnnotations(const QString& imagePath, const QList<AnnotationObject>& annotations, QString* errorMessage) {
    if (annotations.isEmpty()) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("No annotations to append");
        }
        return false;
    }

    const QString jsonPath = jsonPathFromImagePath(imagePath);

    QJsonObject root;
    if (!loadRootObject(jsonPath, imagePath, &root, errorMessage)) {
        return false;
    }

    refreshImageFields(&root, imagePath);

    QJsonArray shapes = root.value(QStringLiteral("shapes")).toArray();
    for (const AnnotationObject& annotation : annotations) {
        if (annotation.label.isEmpty() || annotation.rectPolygonImage.size() != 4) {
            if (errorMessage) {
                *errorMessage = QStringLiteral("Invalid annotation in batch append");
            }
            return false;
        }
        shapes.append(shapeObjectFromAnnotation(annotation));
    }

    root.insert(QStringLiteral("shapes"), shapes);
    return saveRootObject(jsonPath, root, errorMessage);
}

bool AnnotationJsonIO::replaceAnnotations(const QString& imagePath, const QList<AnnotationObject>& annotations,
                                          QString* errorMessage) {
    const QString jsonPath = jsonPathFromImagePath(imagePath);

    QJsonObject root;
    if (!loadRootObject(jsonPath, imagePath, &root, errorMessage)) {
        return false;
    }

    refreshImageFields(&root, imagePath);

    QJsonArray shapes;
    for (const AnnotationObject& annotation : annotations) {
        if (annotation.label.isEmpty() || annotation.rectPolygonImage.size() != 4) {
            if (errorMessage) {
                *errorMessage = QStringLiteral("Invalid annotation in replace");
            }
            return false;
        }
        shapes.append(shapeObjectFromAnnotation(annotation));
    }

    root.insert(QStringLiteral("shapes"), shapes);
    return saveRootObject(jsonPath, root, errorMessage);
}

bool AnnotationJsonIO::clearAnnotations(const QString& imagePath, QString* errorMessage) {
    const QString jsonPath = jsonPathFromImagePath(imagePath);

    QJsonObject root;
    if (!loadRootObject(jsonPath, imagePath, &root, errorMessage)) {
        return false;
    }

    refreshImageFields(&root, imagePath);
    root.insert(QStringLiteral("shapes"), QJsonArray());
    return saveRootObject(jsonPath, root, errorMessage);
}

bool AnnotationJsonIO::removeAnnotationByIndex(const QString& imagePath, int shapeIndex, QString* errorMessage) {
    if (shapeIndex < 0) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("Invalid shape index");
        }
        return false;
    }

    const QString jsonPath = jsonPathFromImagePath(imagePath);

    QJsonObject root;
    if (!loadRootObject(jsonPath, imagePath, &root, errorMessage)) {
        return false;
    }

    QJsonArray shapes = root.value(QStringLiteral("shapes")).toArray();
    if (shapeIndex >= shapes.size()) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("Shape index out of range");
        }
        return false;
    }

    shapes.removeAt(shapeIndex);
    root.insert(QStringLiteral("shapes"), shapes);

    return saveRootObject(jsonPath, root, errorMessage);
}

bool AnnotationJsonIO::updateAnnotationByIndex(const QString& imagePath, int shapeIndex, const AnnotationObject& annotation,
                                               QString* errorMessage) {
    if (shapeIndex < 0) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("Invalid shape index");
        }
        return false;
    }

    const QString jsonPath = jsonPathFromImagePath(imagePath);

    QJsonObject root;
    if (!loadRootObject(jsonPath, imagePath, &root, errorMessage)) {
        return false;
    }

    refreshImageFields(&root, imagePath);

    QJsonArray shapes = root.value(QStringLiteral("shapes")).toArray();
    if (shapeIndex >= shapes.size()) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("Shape index out of range");
        }
        return false;
    }

    shapes.replace(shapeIndex, shapeObjectFromAnnotation(annotation));
    root.insert(QStringLiteral("shapes"), shapes);
    return saveRootObject(jsonPath, root, errorMessage);
}
