#pragma once

#include <QtWidgets/QMainWindow>
#include<qpushbutton.h>
#include "ui_cartoonize.h"
#include<qcamera.h>
#include<qpixmap.h>
#include<qfiledialog.h>
#include<qmessagebox.h>
#include<qscreen.h>
#include<QtMultimediaWidgets/qcameraviewfinder.h>
#include<QtMultimedia/qcameraimagecapture.h>

#include"cartoon_proc.h"


class Cartoonize : public QMainWindow
{
	Q_OBJECT

public:
	Cartoonize(QWidget *parent = Q_NULLPTR);
	void setImage(int, QImage);
	void openFile();
	void saveFile();
	void onImageCapture(int, QImage img);
	void toonify(cv::Mat src);
	void fileToonify();


private:
	Ui::CartoonizeClass ui;
	QString filename;
	QCamera *camera;
};
