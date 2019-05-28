#include"cartoon_proc.h"
using namespace cv;
using namespace std;

void edgesDetection(Mat src, Mat &dst) {
	//Edge.1) Median Filter
	Mat median_result;
	medianBlur(src, median_result, 7);

	//Edge.2) Edge Detection
	Mat edges;
	Canny(median_result, edges, 70, 70 * 3);

	//Edge.3) Edge Filter
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(edges, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE, Point(0, 0));
	for (vector<vector<Point>>::iterator it = contours.begin(); it != contours.end();) {
		if (it->size() < 10)
			it = contours.erase(it);
		else
			++it;
	}
	Mat final_edges = Mat::zeros(edges.size(), CV_8UC1);
	drawContours(final_edges, contours, -1, Scalar(255));

	//Edge.4) Morphological Operations
	Mat dilate_result;
	Mat element = getStructuringElement(MORPH_RECT, Size(2, 2));
	dilate(final_edges, dilate_result, element);

	threshold(dilate_result, dst, 80, 255, THRESH_BINARY_INV);
}

void bilateralSmoothing(Mat src, Mat &dst, int smooth_num, int color_num) {
	smooth_num = smooth_num == 0 ? 7 : smooth_num;
	color_num = color_num == 0 ? 16 : color_num;

	Size size = src.size();
	Size small_size;
	small_size.width = size.width / 2;
	small_size.height = size.height / 2;
	Mat small_img = Mat(small_size, CV_8UC3);
	resize(src, small_img, small_size, 0, 0, INTER_LINEAR);

	Mat tmp = Mat(small_size, CV_8UC3);
	for (int i = 0; i < smooth_num; i++) {
		bilateralFilter(small_img, tmp, 9, 9, 7);
		bilateralFilter(tmp, small_img, 9, 9, 7);
	}

	Mat big_img;
	resize(small_img, big_img, size, 0, 0, INTER_LINEAR);


	//Mat hsv_img;
	//cvtColor(big_img, hsv_img, COLOR_RGB2HSV);
	//for (int i = 0; i < hsv_img.rows; i++)
	//	for (int j = 0; j < hsv_img.cols; j++) {
	//		Vec3b hsv = hsv_img.at<Vec3b>(i, j);

	//		hsv[0] = (hsv[0] / color_num) * color_num;
	//		hsv_img.at<Vec3b>(i, j) = hsv;
	for (int i = 0; i < big_img.rows; i++)
		for (int j = 0; j < big_img.cols; j++) {
			Vec3b rgb = big_img.at<Vec3b>(i, j);
			
			for (int i = 0; i < 3; i++)
				rgb[i] = (rgb[i] / color_num)*color_num;
			big_img.at<Vec3b>(i, j) = rgb;
		}

	big_img.copyTo(dst);
	//cvtColor(hsv_img, dst, COLOR_HSV2RGB);
}

void colorAdjust(Mat src, Mat &dst, float k1, float k2) {
	Mat tmp;
	cvtColor(src, tmp, COLOR_RGB2HSV);
	for (int i = 0; i < tmp.rows; i++)
		for (int j = 0; j < tmp.cols; j++) {
			Vec3b hsv = tmp.at<Vec3b>(i, j);

			float s = hsv[1];
			s = s * k1 >= 255 ? 255 : s * k1;
			hsv[1] = s;

			float v = hsv[2];
			v = v * k2 >= 255 ? 255 : v * k2;
			hsv[2] = v;

			tmp.at<Vec3b>(i, j) = hsv;
		}
	cvtColor(tmp, dst, COLOR_HSV2RGB);
}

void circshift(cv::Mat &A, int shitf_row, int shift_col) {
	int row = A.rows, col = A.cols;
	shitf_row = (row + (shitf_row % row)) % row;
	shift_col = (col + (shift_col % col)) % col;
	cv::Mat temp = A.clone();
	if (shitf_row) {
		temp.rowRange(row - shitf_row, row).copyTo(A.rowRange(0, shitf_row));
		temp.rowRange(0, row - shitf_row).copyTo(A.rowRange(shitf_row, row));
	}
	if (shift_col) {
		temp.colRange(col - shift_col, col).copyTo(A.colRange(0, shift_col));
		temp.colRange(0, col - shift_col).copyTo(A.colRange(shift_col, col));
	}
	return;
}

cv::Mat psf2otf(const cv::Mat &psf, const cv::Size &outSize) {
	cv::Size psfSize = psf.size();
	cv::Mat new_psf = cv::Mat(outSize, CV_64FC2);
	new_psf.setTo(0);
	//new_psf(cv::Rect(0,0,psfSize.width, psfSize.height)).setTo(psf);
	for (int i = 0; i < psfSize.height; i++) {
		for (int j = 0; j < psfSize.width; j++) {
			new_psf.at<cv::Vec2d>(i, j)[0] = psf.at<double>(i, j);
		}
	}

	circshift(new_psf, -1 * int(floor(psfSize.height*0.5)), -1 * int(floor(psfSize.width*0.5)));

	cv::Mat otf;
	cv::dft(new_psf, otf, cv::DFT_COMPLEX_OUTPUT);

	return otf;
}

cv::Mat L0Smoothing(cv::Mat &im8uc3, double lambda, double kappa) {
	// convert the image to double format
	int row = im8uc3.rows, col = im8uc3.cols;
	cv::Mat S;
	im8uc3.convertTo(S, CV_64FC3, 1. / 255.);

	cv::Mat fx(1, 2, CV_64FC1);
	cv::Mat fy(2, 1, CV_64FC1);
	fx.at<double>(0) = 1; fx.at<double>(1) = -1;
	fy.at<double>(0) = 1; fy.at<double>(1) = -1;

	cv::Size sizeI2D = im8uc3.size();
	cv::Mat otfFx = psf2otf(fx, sizeI2D);
	cv::Mat otfFy = psf2otf(fy, sizeI2D);

	cv::Mat Normin1[3];
	cv::Mat single_channel[3];
	cv::split(S, single_channel);
	for (int k = 0; k < 3; k++) {
		cv::dft(single_channel[k], Normin1[k], cv::DFT_COMPLEX_OUTPUT);
	}
	cv::Mat Denormin2(row, col, CV_64FC1);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			cv::Vec2d &c1 = otfFx.at<cv::Vec2d>(i, j), &c2 = otfFy.at<cv::Vec2d>(i, j);
			Denormin2.at<double>(i, j) = sqr(c1[0]) + sqr(c1[1]) + sqr(c2[0]) + sqr(c2[1]);
		}
	}

	double beta = 2.0*lambda;
	double betamax = 1e5;

	while (beta < betamax) {
		cv::Mat Denormin = 1.0 + beta * Denormin2;

		// h-v subproblem
		cv::Mat dx[3], dy[3];
		for (int k = 0; k < 3; k++) {
			cv::Mat shifted_x = single_channel[k].clone();
			circshift(shifted_x, 0, -1);
			dx[k] = shifted_x - single_channel[k];

			cv::Mat shifted_y = single_channel[k].clone();
			circshift(shifted_y, -1, 0);
			dy[k] = shifted_y - single_channel[k];
		}
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				double val =
					sqr(dx[0].at<double>(i, j)) + sqr(dy[0].at<double>(i, j)) +
					sqr(dx[1].at<double>(i, j)) + sqr(dy[1].at<double>(i, j)) +
					sqr(dx[2].at<double>(i, j)) + sqr(dy[2].at<double>(i, j));

				if (val < lambda / beta) {
					dx[0].at<double>(i, j) = dx[1].at<double>(i, j) = dx[2].at<double>(i, j) = 0.0;
					dy[0].at<double>(i, j) = dy[1].at<double>(i, j) = dy[2].at<double>(i, j) = 0.0;
				}
			}
		}

		// S subproblem
		for (int k = 0; k < 3; k++) {
			cv::Mat shift_dx = dx[k].clone();
			circshift(shift_dx, 0, 1);
			cv::Mat ddx = shift_dx - dx[k];

			cv::Mat shift_dy = dy[k].clone();
			circshift(shift_dy, 1, 0);
			cv::Mat ddy = shift_dy - dy[k];
			cv::Mat Normin2 = ddx + ddy;
			cv::Mat FNormin2;
			cv::dft(Normin2, FNormin2, cv::DFT_COMPLEX_OUTPUT);
			cv::Mat FS = Normin1[k] + beta * FNormin2;
			for (int i = 0; i < row; i++) {
				for (int j = 0; j < col; j++) {
					FS.at<cv::Vec2d>(i, j)[0] /= Denormin.at<double>(i, j);
					FS.at<cv::Vec2d>(i, j)[1] /= Denormin.at<double>(i, j);
				}
			}
			cv::Mat ifft;
			cv::idft(FS, ifft, cv::DFT_SCALE | cv::DFT_COMPLEX_OUTPUT);
			for (int i = 0; i < row; i++) {
				for (int j = 0; j < col; j++) {
					single_channel[k].at<double>(i, j) = ifft.at<cv::Vec2d>(i, j)[0];
				}
			}
		}
		beta *= kappa;
		std::cout << '.';
	}
	cv::merge(single_channel, 3, S);

	S.convertTo(S, CV_8U, 255);
	return S;
}

Mat QImageToMat(QImage image)
{
	cv::Mat mat;
	switch (image.format())
	{
	case QImage::Format_ARGB32:
	case QImage::Format_RGB32:
	case QImage::Format_ARGB32_Premultiplied:
		mat = cv::Mat(image.height(), image.width(), CV_8UC4, (void*)image.constBits(), image.bytesPerLine());
		break;
	case QImage::Format_RGB888:
		mat = cv::Mat(image.height(), image.width(), CV_8UC3, (void*)image.constBits(), image.bytesPerLine());
		cv::cvtColor(mat, mat, COLOR_BGR2RGB);
		break;
	case QImage::Format_Indexed8:
		mat = cv::Mat(image.height(), image.width(), CV_8UC1, (void*)image.constBits(), image.bytesPerLine());
		break;
	}
	return mat;
}

QImage MatToQImage(const cv::Mat& mat)
{
	// 8-bits unsigned, NO. OF CHANNELS = 1  
	if (mat.type() == CV_8UC1)
	{
		QImage image(mat.cols, mat.rows, QImage::Format_Indexed8);
		// Set the color table (used to translate colour indexes to qRgb values)  
		image.setColorCount(256);
		for (int i = 0; i < 256; i++)
		{
			image.setColor(i, qRgb(i, i, i));
		}
		// Copy input Mat  
		uchar *pSrc = mat.data;
		for (int row = 0; row < mat.rows; row++)
		{
			uchar *pDest = image.scanLine(row);
			memcpy(pDest, pSrc, mat.cols);
			pSrc += mat.step;
		}
		return image;
	}
	// 8-bits unsigned, NO. OF CHANNELS = 3  
	else if (mat.type() == CV_8UC3)
	{
		// Copy input Mat  
		const uchar *pSrc = (const uchar*)mat.data;
		// Create QImage with same dimensions as input Mat  
		QImage image(pSrc, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
		return image.rgbSwapped();
	}
	else if (mat.type() == CV_8UC4)
	{
		qDebug() << "CV_8UC4";
		// Copy input Mat  
		const uchar *pSrc = (const uchar*)mat.data;
		// Create QImage with same dimensions as input Mat  
		QImage image(pSrc, mat.cols, mat.rows, mat.step, QImage::Format_ARGB32);
		return image.copy();
	}
	else
	{
		qDebug() << "ERROR: Mat could not be converted to QImage.";
		return QImage();
	}
}