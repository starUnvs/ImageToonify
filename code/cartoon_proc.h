#pragma once
#include<opencv.hpp>
#include<opencv2/core.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>
#include<qcamera.h>
#include<vector>

void edgesDetection(cv::Mat src, cv::Mat & dst);
void bilateralSmoothing(cv::Mat src, cv::Mat & dst, int smooth_num = 7, int color_num = 16);

void colorAdjust(cv::Mat src, cv::Mat & dst, float k = 1.3, float k2 = 1.3);

cv::Mat psf2otf(const cv::Mat &psf, const cv::Size &outSize);
void circshift(cv::Mat &A, int shift_row, int shift_col);
cv::Mat L0Smoothing(cv::Mat & im8uc3, double lambda = 2e-2, double kappa = 2.0);

cv::Mat QImageToMat(QImage image);

QImage MatToQImage(const cv::Mat & mat);

template<typename T>
T sqr(const T x) { return x * x; }