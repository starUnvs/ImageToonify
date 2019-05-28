#include "cartoonize.h"

Cartoonize::Cartoonize(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);

	camera = new QCamera(this);
	QCameraViewfinder *viewfinder = new QCameraViewfinder(this);
	ui.horizontalLayout->addWidget(viewfinder);
	camera->setViewfinder(viewfinder);
	QCameraImageCapture *capture = new QCameraImageCapture(camera, this);

	viewfinder->show();
	camera->setCaptureMode(QCamera::CaptureStillImage);
	camera->start();

	connect(ui.pushButton, SIGNAL(clicked()), capture, SLOT(capture()));
	connect(capture, &QCameraImageCapture::imageCaptured, this, &Cartoonize::onImageCapture);
	connect(ui.pushButton_2, &QPushButton::clicked, this, &Cartoonize::openFile);
	connect(ui.pushButton_3, &QPushButton::clicked, this, &Cartoonize::saveFile);
	connect(ui.pushButton_4, &QPushButton::clicked, this, &Cartoonize::fileToonify);
}



void Cartoonize::setImage(int index, QImage image)
{
	QPixmap pixmap = QPixmap::fromImage(image).scaled(ui.label->size());
	if (index == 1)
		ui.label->setPixmap(pixmap);
	else if (index == 2)
		ui.label_2->setPixmap(pixmap);
}

void Cartoonize::openFile()
{
	filename = QFileDialog::getOpenFileName(this, tr("Ñ¡ÔñÍ¼Ïñ"), "", tr("Images (*.png *.bmp *.jpg)"));
	QImage img;
	if (filename.isEmpty())
		return;
	else{
		if (!(img.load(filename))){ //¼ÓÔØÍ¼Ïñ
			QMessageBox::information(this, tr("´ò¿ªÍ¼ÏñÊ§°Ü"), tr("´ò¿ªÍ¼ÏñÊ§°Ü!"));
			return;
		}
		QStringList list = filename.split('/');
		ui.label_8->setText(list.last());
	}
	fileToonify();
}

void Cartoonize::saveFile()
{
	QString filename1 = QFileDialog::getSaveFileName(this, tr("Save Image"), "", tr("Images (*.jpg)")); //Ñ¡ÔñÂ·¾¶
	QScreen *screen = QGuiApplication::primaryScreen();
	screen->grabWindow(ui.label->winId()).save(filename1);
	QString filename2 = QFileDialog::getSaveFileName(this, tr("Save Image"), "", tr("Images (*.jpg)")); //Ñ¡ÔñÂ·¾¶
	screen->grabWindow(ui.label_2->winId()).save(filename2);
}

void Cartoonize::onImageCapture(int, QImage img)
{
	camera->stop();
	cv::VideoCapture cap(0);
	cv::Mat src;
	cap >> src;
	cap.~VideoCapture();
	camera->start();

	toonify(src);
}

void Cartoonize::toonify(cv::Mat src)
{
	cv::Mat result1, result2;

	int smooth_level = ui.smooth_level->text().toInt(), color_level = ui.color_level->text().toInt();
	bilateralSmoothing(src, result1, smooth_level, color_level);
	colorAdjust(result1, result1);

	if (ui.enable_L0->isChecked()) {
		double lambda = ui.lambda->text().toDouble();
		double kappa = ui.kappa->text().toDouble();
		lambda = lambda == 0 ? 0.2 : lambda;
		kappa = kappa == 0 ? 10 : kappa;
		result2 = L0Smoothing(src,lambda,kappa);
		colorAdjust(result2, result2);
	}

	cv::Mat dst1, dst2;
	if (ui.enable_edge->isChecked()) {
		cv::Mat edges;
		edgesDetection(src, edges);
		
		result1.copyTo(dst1, edges);
		if (ui.enable_L0->isChecked())
			result2.copyTo(dst2, edges);
	}
	else {
		dst1 = result1;
		dst2 = result2;
	}

	setImage(2, MatToQImage(dst2));
	setImage(1, MatToQImage(dst1));
}

void Cartoonize::fileToonify()
{
	cv::Mat src = cv::imread(filename.toStdString());
	toonify(src);
}
