#include "Image.h"
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include "Kernel.h"

Image::Image()
{
}

Image::Image(const unsigned height, const unsigned width)
	: mImage(height, width, CV_32FC1)
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
		for (auto v = 0; v <= 2; v++) {
			temp = temp + (kernel.at<int>(u, v) * image.at<cv::Vec3b>(i + (u - 1), j + (v - 1))[color]);
		}
	}
	temp = abs(temp);
	if (temp > 255) {
		return 255;
	}
	else
		return temp;
}

Image Image::convolution(const Kernel& kernel) const
{
	Image image_convoluee;
	image_convoluee.mImage = cv::Mat(mImage.size().height - 2, mImage.size().width - 2, mImage.type());

	for (auto i = 1; i < mImage.size().height - 1; ++i) {
		for (auto j = 1; j < mImage.size().width - 1; ++j) {
			for (auto k = 0; k < mImage.channels(); k++) {
				image_convoluee.mImage.at<cv::Vec3b>(i - 1, j - 1)[k] = conv(mImage, kernel._Mat(), i, j, k);
			}
		}
	}

	return std::move(image_convoluee);
}

static float calcul_moyenne(const unsigned a, const unsigned b, const unsigned c) {
	return float((a + b + c)) / 3.f;
}

std::pair<Image, Image> Image::bidirectionalGradient(const Image& i1, const Image& i2)
{
	auto gradient = std::make_pair(Image(i1.height(), i1.width()), Image(i2.height(), i2.width()));

	for (auto j = 0; j < i1._Mat().size().width; ++j) {
		for (auto i = 0; i < i1._Mat().size().height; ++i) {
			gradient.first.mImage.at<float>(i, j) = sqrt(std::pow(calcul_moyenne(i1._Mat().at<cv::Vec3b>(i, j)[0],
				i1._Mat().at<cv::Vec3b>(i, j)[1],
				i1._Mat().at<cv::Vec3b>(i, j)[2]
			),
				2) +
				std::pow(calcul_moyenne(i2._Mat().at<cv::Vec3b>(i, j)[0],
					i2._Mat().at<cv::Vec3b>(i, j)[1],
					i2._Mat().at<cv::Vec3b>(i, j)[2]
				), 2)
			);
			gradient.second.mImage.at<float>(i, j) = cvFastArctan(calcul_moyenne(i2._Mat().at<cv::Vec3b>(i, j)[0],
				i2._Mat().at<cv::Vec3b>(i, j)[1],
				i2._Mat().at<cv::Vec3b>(i, j)[2]
			),
				calcul_moyenne(i1._Mat().at<cv::Vec3b>(i, j)[0],
					i1._Mat().at<cv::Vec3b>(i, j)[1],
					i1._Mat().at<cv::Vec3b>(i, j)[2]
				)
			);
		}
	}
	return std::move(gradient);
}

const cv::Mat& Image::_Mat() const
{
	return mImage;
}

unsigned Image::width() const
{
	return mImage.size().width;
}

unsigned Image::height() const
{
	return mImage.size().height;
}
