QT += core gui widgets

CONFIG += c++17
win32-msvc*: QMAKE_CXXFLAGS += /utf-8
TEMPLATE = app
TARGET = SemiAutoAnnoSoftware
DESTDIR = E:/autolabelproject/BIN

DEFINES += NOMINMAX

SOURCES += \
    main.cpp \
    app/MainWindow.cpp \
    widgets/ImageAnnotateWidget.cpp \
    dialogs/LabelSelectDialog.cpp \
    dialogs/AddLabelDialog.cpp \
    inference/SamInferenceBridge.cpp \
    data/AnnotationJsonIO.cpp \
    data/LabelConfigIO.cpp

HEADERS += \
    app/AppTypes.h \
    app/MainWindow.h \
    widgets/ImageAnnotateWidget.h \
    dialogs/LabelSelectDialog.h \
    dialogs/AddLabelDialog.h \
    inference/SamTypes.h \
    inference/SamInferenceBridge.h \
    data/AnnotationJsonIO.h \
    data/LabelConfigIO.h

INCLUDEPATH += \
    $$PWD \
    $$PWD/app \
    $$PWD/widgets \
    $$PWD/dialogs \
    $$PWD/inference \
    $$PWD/data \
    $$PWD/../SamBaseLib \
    $$PWD/../TrtSam3Lib

SAM_LIB_DIR = F:/ytprojectv2alln/BINX64_YoloTraingrayV2

LIBS += \
    -L$${SAM_LIB_DIR} \
    -lSamBaseLib \
    -lTrtSam3Lib
