#include "data/LabelConfigIO.h"

#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

namespace {
void ensureColorSize(LabelConfig* config) {
    if (config == nullptr) {
        return;
    }
    while (config->colorDefine.size() < config->nameList.size()) {
        config->colorDefine.push_back(0x00FF00);
    }
    while (config->colorDefine.size() > config->nameList.size()) {
        config->colorDefine.removeLast();
    }
}
}

bool LabelConfigIO::loadConfig(const QString& configPath, LabelConfig* config, QString* errorMessage) {
    if (config == nullptr) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("Internal error: null config");
        }
        return false;
    }

    QFile file(configPath);
    if (!file.exists()) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("Config not found: %1").arg(configPath);
        }
        return false;
    }

    if (!file.open(QIODevice::ReadOnly)) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("Failed to open config: %1").arg(configPath);
        }
        return false;
    }

    const QByteArray data = file.readAll();
    file.close();

    QJsonParseError parseError{};
    const QJsonDocument doc = QJsonDocument::fromJson(data, &parseError);
    if (parseError.error != QJsonParseError::NoError || !doc.isObject()) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("Invalid config json: %1").arg(configPath);
        }
        return false;
    }

    const QJsonObject root = doc.object();
    config->infoSet = root.value(QStringLiteral("InfoSet")).toString();

    config->nameList.clear();
    for (const QJsonValue& v : root.value(QStringLiteral("NameList")).toArray()) {
        if (v.isString()) {
            config->nameList.push_back(v.toString());
        }
    }

    config->colorDefine.clear();
    for (const QJsonValue& v : root.value(QStringLiteral("ClorDefine")).toArray()) {
        config->colorDefine.push_back(v.toInt(0x00FF00));
    }

    ensureColorSize(config);
    return true;
}

bool LabelConfigIO::saveConfig(const QString& configPath, const LabelConfig& config, QString* errorMessage) {
    LabelConfig normalized = config;
    ensureColorSize(&normalized);

    QJsonObject root;
    root.insert(QStringLiteral("InfoSet"), normalized.infoSet);

    QJsonArray nameList;
    for (const QString& n : normalized.nameList) {
        nameList.append(n);
    }
    root.insert(QStringLiteral("NameList"), nameList);

    QJsonArray colorList;
    for (int c : normalized.colorDefine) {
        colorList.append(c);
    }
    root.insert(QStringLiteral("ClorDefine"), colorList);

    QFile file(configPath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("Failed to write config: %1").arg(configPath);
        }
        return false;
    }

    file.write(QJsonDocument(root).toJson(QJsonDocument::Indented));
    file.close();
    return true;
}

bool LabelConfigIO::addLabel(LabelConfig* config, const QString& labelName, int colorValue, QString* errorMessage) {
    if (config == nullptr) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("Internal error: null config");
        }
        return false;
    }

    const QString normalized = labelName.trimmed();
    if (normalized.isEmpty()) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("Label name is empty");
        }
        return false;
    }

    if (config->nameList.contains(normalized)) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("Label already exists");
        }
        return false;
    }

    config->nameList.push_back(normalized);
    config->colorDefine.push_back(colorValue);
    ensureColorSize(config);
    return true;
}

bool LabelConfigIO::removeLabel(LabelConfig* config, int index, QString* errorMessage) {
    if (config == nullptr) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("Internal error: null config");
        }
        return false;
    }

    if (index < 0 || index >= config->nameList.size()) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("Label index out of range");
        }
        return false;
    }

    config->nameList.removeAt(index);
    if (index >= 0 && index < config->colorDefine.size()) {
        config->colorDefine.removeAt(index);
    }
    ensureColorSize(config);
    return true;
}
