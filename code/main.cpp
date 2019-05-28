#include "cartoonize.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	Cartoonize w;
	w.show();
	return a.exec();
}
