#include "Image.h"
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include "Kernel.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <cstdarg>
#include <opencv2/highgui/highgui.hpp>

#ifdef WIN32
#include <corecrt_math_defines.h>
#endif

Image::Image()
{
}

Image::Image(const unsigned height, const unsigned width)
	: mImage(height, width, CV_32F)
{
}

bool Image::readFromFile(const char* path)
{
	mImage = std::move(cv::imread(path));
	if (!mImage.data) {
		std::cerr << "Error : Can't read image at path : " << path << std::endl;
		return false;
	}
	return true;
}

int conv(const cv::Mat& image, const cv::Mat& kernel, const int i, const int j, const int color) {
	// Problème sur les bords de l'image
	auto temp = 0;
	for (auto u = 0; u <= 2; u++) {
		for (auto v = 0; v <=2; v++) {
			temp = temp + (kernel.at<int>(u, v) * image.at<cv::Vec3b>(i + (u - 1), j + (v - 1))[color]);
		}
	}
	temp = abs(temp);
	if (temp > 255) {
		return 255;
	}
	return temp;
}

Image Image::convolution(const Kernel& kernel) const
{
	Image image_convoluee;
	image_convoluee.mImage = cv::Mat(mImage.size().height - 2, mImage.size().width - 2, mImage.type());

	// Pour chaque canal de chaque pixel de l'image, on applique le produit de convolution avec le noyau
	for (auto i = 1; i < mImage.size().height - 1; ++i) {
		for (auto j = 1; j < mImage.size().width - 1; ++j) {
			for (auto k = 0; k < mImage.channels(); k++) {
				image_convoluee.mImage.at<cv::Vec3b>(i - 1, j - 1)[k] = conv(mImage, kernel._Mat(), i, j, k);
			}
		}
	}

	return std::move(image_convoluee);
}

Image Image::toGray() const
{
	cv::Mat gray_mat;
	cvtColor(this->mImage, gray_mat, cv::COLOR_RGB2GRAY);

	Image gray_img;
	gray_img.mImage = gray_mat;

	return std::move(gray_img);
	
}

void Image::convertToFloat () {
    this->mImage.convertTo(this->mImage, CV_32F);
}

template< typename T>
static double mean(std::initializer_list<T> list)
{
	double mean = 0;
	for (auto& elmt : list)
		mean += double(elmt);
	return mean / double(list.size());
}

std::pair<Image, Image> Image::bidirectionalGradient(const Image& i1, const Image& i2)
{
	auto gradient = std::make_pair(Image(i1.height(), i1.width()), Image(i2.height(), i2.width()));

    for (auto i = 0; i < i1._Mat().size().height; ++i) {
        for (auto j = 0; j < i1._Mat().size().width; ++j) {
			gradient.first.mImage.at<float>(i, j) = sqrt(std::pow(mean({ i1._Mat().at<cv::Vec3b>(i, j)[0],
				i1._Mat().at<cv::Vec3b>(i, j)[1],
				i1._Mat().at<cv::Vec3b>(i, j)[2] }
			),
				2) +
				std::pow(mean({ i2._Mat().at<cv::Vec3b>(i, j)[0],
					i2._Mat().at<cv::Vec3b>(i, j)[1],
					i2._Mat().at<cv::Vec3b>(i, j)[2] }
				), 2)
			);
			gradient.second.mImage.at<float>(i, j) = cvFastArctan(mean({ i2._Mat().at<cv::Vec3b>(i, j)[0],
				i2._Mat().at<cv::Vec3b>(i, j)[1],
				i2._Mat().at<cv::Vec3b>(i, j)[2] }
			),
				mean({ i1._Mat().at<cv::Vec3b>(i, j)[0],
					i1._Mat().at<cv::Vec3b>(i, j)[1],
					i1._Mat().at<cv::Vec3b>(i, j)[2] }
				)
			);
		}
	}
	return std::move(gradient);
}


std::pair<Image, Image> Image::multidirectionalDirection(const Image& distance, const Image& gray0,
                                                         const Image& gray1, const Image& gray2, const Image& gray3)
{
	Image result(distance.height(), distance.width());
	Image direction_color(distance.height(), distance.width());
	direction_color.mImage = cv::Mat(distance.height(), distance.width(), CV_8UC3);

	for (auto i = 0; i < distance.height(); ++i) {
		for (auto j = 0; j < distance.width(); ++j) {
			if (distance.mImage.at<float>(i, j) == 0) {
				result.mImage.at<float>(i, j) = 0;
				direction_color.mImage.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
			}
			else if (distance.mImage.at<float>(i, j) == gray0.mImage.at<uchar>(i, j)) {
				result.mImage.at<float>(i, j) = 0;
				direction_color.mImage.at<cv::Vec3b>(i, j) = cv::Vec3b(200, 0, 0);
			}
			else if (distance.mImage.at<float>(i, j) == gray1.mImage.at<uchar>(i, j)) {
				result.mImage.at<float>(i, j) = 1 * (M_PI_4);
				direction_color.mImage.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 200, 0);
			}
			else if (distance.mImage.at<float>(i, j) == gray2.mImage.at<uchar>(i, j)) {
				result.mImage.at<float>(i, j) = 2 * (M_PI_4);
				direction_color.mImage.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 200);
			}
			else {
				result.mImage.at<float>(i, j) = 3 * (M_PI_4);
				direction_color.mImage.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255);
			}
		}
	}

	return std::move(std::make_pair(result, direction_color));
}

Image Image::thresholding(const Image& source, const float& threshold) {
    Image result(source.height(), source.width());
    
    for (int i=0; i < source.height(); ++i) {
        for (int j=0; j < source.width(); ++j) {
            if (source.mImage.at<float>(i,j) < threshold * 255) {
                result.mImage.at<float>(i,j) = 0;
            } else {
                result.mImage.at<float>(i,j) = 255;
            }
        }
    }
    return result;
}

Image Image::max(const Image& i0, const Image& i1)
{
	Image result;
	result.mImage = cv::max(i0.mImage, i1.mImage);
	return result;
}

const cv::Mat& Image::_Mat() const
{
	return mImage;
}

const float& Image::operator()(const unsigned i, const unsigned j) const
{
	return mImage.at<float>(i, j);
}

void Image::show(const char* name) const
{
	cv::namedWindow(name, 1);
	cv::imshow(name, mImage);
}

unsigned Image::width() const
{
	return mImage.size().width;
}

unsigned Image::height() const
{
	return mImage.size().height;
}
