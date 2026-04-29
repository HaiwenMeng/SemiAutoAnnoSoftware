#ifndef AUTOLABELPROJECT_DATA_LABELCONFIGIO_H
#define AUTOLABELPROJECT_DATA_LABELCONFIGIO_H

#include <QString>

#include "app/AppTypes.h"

class LabelConfigIO {
public:
    static bool loadConfig(const QString& configPath, LabelConfig* config, QString* errorMessage = nullptr);
    static bool saveConfig(const QString& configPath, const LabelConfig& config, QString* errorMessage = nullptr);

    static bool addLabel(LabelConfig* config, const QString& labelName, int colorValue, QString* errorMessage = nullptr);
    static bool removeLabel(LabelConfig* config, int index, QString* errorMessage = nullptr);
};

#endif // AUTOLABELPROJECT_DATA_LABELCONFIGIO_H
