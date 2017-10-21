#include "Image.h"
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include "Kernel.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <cstdarg>
#include <opencv2/highgui/highgui.hpp>
#include <queue>
#include <list>

#if defined(_WIN32) || defined(WIN32)
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
	for (auto u = 0; u < kernel.size().width; u++) {
		for (auto v = 0; v < kernel.size().height; v++) {
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
	Image gray_img;
	cvtColor(this->mImage, gray_img.mImage, cv::COLOR_RGB2GRAY);
	return std::move(gray_img);
}

void Image::convertToFloat () {

	mImage.clone().convertTo(mImage, CV_32F);
}

cv::Mat* Image::_MatPtr()
{
	return &mImage;
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

    std::cout << distance._Mat().type();
    std::cout << gray0._Mat().type();
	for (auto i = 0; i < distance.height(); ++i) {
		for (auto j = 0; j < distance.width(); ++j) {
			if (distance.mImage.at<float>(i, j) == 0) {
				result.mImage.at<float>(i, j) = 0.f;
				direction_color.mImage.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
			}
			else if (distance.mImage.at<float>(i, j) == gray0.mImage.at<float>(i, j)) {
				result.mImage.at<float>(i, j) = 0.f;
				direction_color.mImage.at<cv::Vec3b>(i, j) = cv::Vec3b(200, 0, 0);
			}
			else if (distance.mImage.at<float>(i, j) == gray1.mImage.at<float>(i, j)) {
				result.mImage.at<float>(i, j) = M_PI_4;
				direction_color.mImage.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 200, 0);
			}
			else if (distance.mImage.at<float>(i, j) == gray2.mImage.at<float>(i, j)) {
				result.mImage.at<float>(i, j) = M_PI_2;
				direction_color.mImage.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 200);
			}
			else {
				result.mImage.at<float>(i, j) = 3.f * M_PI_4;
				direction_color.mImage.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255);
			}
		}
	}

	return std::move(std::make_pair(result, direction_color));
}

float Image::calcLocalThreshold(const Image& source, int size, int i, int j) {
    cv::Mat patch;
    cv::getRectSubPix(source._Mat(), cv::Size(size,size), cv::Point(j,i), patch);
    cv::Scalar mean, stddev, result;
    cv::meanStdDev(patch, mean, stddev, cv::Mat());
    result =  mean;
    return result.val[0];
}

Image Image::localThresholding(const Image& source, const int size) {
    float localThreshold;
    Image result(source.height(), source.width());
    
    for (int i=0; i < source.height(); ++i) {
        for (int j=0; j < source.width(); ++j) {
            localThreshold = calcLocalThreshold(source, size, i, j);
            if (source.mImage.at<float>(i,j) > localThreshold) {
                result.mImage.at<float>(i,j) = 255;
            } else {
                result.mImage.at<float>(i,j) = 0;
            }
        }
    }
    return std::move(result);
}

Image Image::thresholding(const Image& source, const float& threshold) {
    Image result(source.height(), source.width());
    
    for (int i=0; i < source.height(); ++i) {
        for (int j=0; j < source.width(); ++j) {
            if (source.mImage.at<float>(i,j) < threshold) {
                result.mImage.at<float>(i,j) = 0;
            } else {
                result.mImage.at<float>(i,j) = 255;
            }
        }
    }
    return std::move(result);
}

Image Image::thresholdingLow(const Image& source, const Image& temp, const float& threshold) {
    Image result(source.height(), source.width());
    
    int count;
    int H = source.height();
    int W = source.width();
    for (int i=0; i < H; ++i) {
        for (int j=0; j < W; ++j) {
            if (source.mImage.at<float>(i,j) < threshold)
                result.mImage.at<float>(i,j) = 0;
            else {
                count = 0;
                for (int k = std::max(0,i-1); k < std::min(i+1, H); ++k) {
                    for (int l = std::max(0,j-1); l < std::min(j+1, W); ++l) {
                        if (temp.mImage.at<float>(k,l) == 255)
                            count++;
                    }
                }
                if (count >= 0)
                    result.mImage.at<float>(i,j) = 255;
            }
        }
    }
    return std::move(result);
}

Image Image::thresholdingHysteresis(const Image& source, const float& thresholdHigh, const float& thresholdLow) {
    Image result(source.height(), source.width());
    result.thresholding(source, thresholdHigh);
    result = thresholdingLow(source, result, thresholdLow);
    
    return result;
}

Image Image::thinningMulti(const Image& source, const Image& grad, const Image& dir) {
    Image result(source.height(), source.width());

    for (int i=1; i < grad.height() -1; ++i) {
        for (int j=1; j < grad.width() -1; ++j) {
            if (source.mImage.at<float>(i,j) == 255) {
                double angle = fmod ((dir.mImage.at<float>(i,j)) + M_PI, M_PI);
                if (angle >= 1./2. * M_PI_4 && angle < 3./2. * M_PI_4) {
                    //zone autour de PI/4
                    if (grad.mImage.at<float>(i,j) > grad.mImage.at<float>(i-1,j+1) &&
                        grad.mImage.at<float>(i,j) > grad.mImage.at<float>(i+1,j-1))
                        result.mImage.at<float>(i,j) = 255;
                    else
                        result.mImage.at<float>(i,j) = 0;
                } else if (angle >= 3./2. * M_PI_4 && angle < 5./2. * M_PI_4) {
                    //zone autour de PI/2
                    if (grad.mImage.at<float>(i,j) > grad.mImage.at<float>(i-1,j) &&
                        grad.mImage.at<float>(i,j) > grad.mImage.at<float>(i+1,j))
                        result.mImage.at<float>(i,j) = 255;
                    else
                        result.mImage.at<float>(i,j) = 0;
                } else if (angle >= 5./2. * M_PI_4 && angle < 7./2. * M_PI_4) {
                    //zone autour de 3PI/4
                    if (grad.mImage.at<float>(i,j) > grad.mImage.at<float>(i+1,j+1) &&
                        grad.mImage.at<float>(i,j) > grad.mImage.at<float>(i-1,j-1))
                        result.mImage.at<float>(i,j) = 255;
                    else
                        result.mImage.at<float>(i,j) = 0;
                } else {
                    //zone autour de 0 et de PI
                    if (grad.mImage.at<float>(i,j) > grad.mImage.at<float>(i,j-1) &&
                        grad.mImage.at<float>(i,j) > grad.mImage.at<float>(i,j+1))
                        result.mImage.at<float>(i,j) = 255;
                    else
                        result.mImage.at<float>(i,j) = 0;
                }
            } else {
                result.mImage.at<float>(i,j) = 0;
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

Image Image::closure(const Image& source, const Image& norme)
{
	const float seuil = 255 * 2 + 222;
	Image result = source;

	for (unsigned i = 2; i < source.height() - 2; ++i)
		for (unsigned j = 2; j < source.width() - 2; ++j)
		{
			if (source.mImage.at<float>(i, j) == 255.0f)
			{
				std::vector<std::tuple<unsigned, unsigned>> candidates;

				std::tuple<unsigned, unsigned, unsigned, unsigned> t = std::make_tuple(0, 0, 0, 0);
				// top
				if (source.mImage.at<float>(i, j - 1) == 255.0f)
				{
					const float s = norme.mImage.at<float>(i, j) + norme.mImage.at<float>(i, j - 1);
					const double sum1 = s + norme.mImage.at<float>(i - 1, j - 2); // top top left
					const double sum2 = s + norme.mImage.at<float>(i, j - 2); // top top
					const double sum3 = s + norme.mImage.at<float>(i + 1, j - 2); // top top right

					if (sum1 > sum2)
					{
						if (sum1 > sum3)
						{
							if (sum1 > seuil)
								t = (std::make_tuple(i - 1, j - 2, i, j - 1));
						}
						else if (sum3 > seuil)
							t = (std::make_tuple(i + 1, j - 2, i, j - 1));
					}
					else
					{
						if (sum2 > sum3)
						{
							if (sum2 > seuil)
								t = (std::make_tuple(i, j - 2, i, j - 1));
						}
						else if (sum3 > seuil)
							t = (std::make_tuple(i + 1, j - 2, i, j - 1));
					}

					if (std::get<0>(t) != 0 || std::get<1>(t) != 0 || std::get<2>(t) != 0 || std::get<3>(t) != 0)
					{
						auto k = std::get<0>(t);
						auto l = std::get<1>(t);

						while (k < source.height() && l < source.width())
						{
							if (source.mImage.at<float>(k, l) == 255.0f)
							{
								for (auto& candidate : candidates)
								{
									result.mImage.at<float>(std::get<0>(candidate), std::get<1>(candidate)) = 255.0f;
								}

								break;
							}

							candidates.push_back(std::make_tuple(std::get<0>(t), std::get<1>(t)));

							t = std::make_tuple(k + k - std::get<2>(t), l + l - std::get<3>(t), k, l);

							k = std::get<0>(t);
							l = std::get<1>(t);
						}
					}
					candidates.clear();
				}

				t = std::make_tuple(0, 0, 0, 0);
				// top right
				if (source.mImage.at<float>(i + 1, j - 1) == 255.0f)
				{
					const float s = norme.mImage.at<float>(i, j) + norme.mImage.at<float>(i + 1, j - 1);
					const double sum1 = s + norme.mImage.at<float>(i + 1, j - 2); // top right top
					const double sum2 = s + norme.mImage.at<float>(i + 1, j - 2); // top right top right
					const double sum3 = s + norme.mImage.at<float>(i + 2, j - 1); // top right right

					if (sum1 > sum2)
					{
						if (sum1 > sum3)
						{
							if (sum1 > seuil)
								t = (std::make_tuple(i + 1, j - 2, i + 1, j - 1));
						}
						else if (sum3 > seuil)
							t = (std::make_tuple(i + 2, j - 1, i + 1, j - 1));
					}
					else
					{
						if (sum2 > sum3)
						{
							if (sum2 > seuil)
								t = (std::make_tuple(i + 1, j - 2, i + 1, j - 1));
						}
						else if (sum3 > seuil)
							t = (std::make_tuple(i + 2, j - 1, i + 1, j - 1));
					}
					if (std::get<0>(t) != 0 || std::get<1>(t) != 0 || std::get<2>(t) != 0 || std::get<3>(t) != 0)
					{
						auto k = std::get<0>(t);
						auto l = std::get<1>(t);

						while (k < source.height() && l < source.width())
						{
							if (source.mImage.at<float>(k, l) == 255.0f)
							{
								for (auto& candidate : candidates)
								{
									result.mImage.at<float>(std::get<0>(candidate), std::get<1>(candidate)) = 255.0f;
								}

								break;
							}

							candidates.push_back(std::make_tuple(std::get<0>(t), std::get<1>(t)));

							t = std::make_tuple(k + k - std::get<2>(t), l + l - std::get<3>(t), k, l);

							k = std::get<0>(t);
							l = std::get<1>(t);
						}
					}
					candidates.clear();
				}

				t = std::make_tuple(0, 0, 0, 0);
				// right
				if (source.mImage.at<float>(i + 1, j) == 255.0f)
				{
					const float s = norme.mImage.at<float>(i, j) + norme.mImage.at<float>(i + 1, j);
					const double sum1 = s + norme.mImage.at<float>(i + 2, j - 1); // right top right
					const double sum2 = s + norme.mImage.at<float>(i + 2, j); // right right
					const double sum3 = s + norme.mImage.at<float>(i + 2, j + 1); // right bottom right
					if (sum1 > sum2)
					{
						if (sum1 > sum3)
						{
							if (sum1 > seuil)
								t = (std::make_tuple(i + 2, j - 1, i + 1, j));
						}
						else if (sum3 > seuil)
							t = (std::make_tuple(i + 2, j + 1, i + 1, j));
					}
					else if (sum2 > sum3)
					{
						if (sum2 > seuil)
							t = (std::make_tuple(i + 2, j, i + 1, j));
						else if (sum3 > seuil)
							t = (std::make_tuple(i + 2, j + 1, i + 1, j));
					}
					if (std::get<0>(t) != 0 || std::get<1>(t) != 0 || std::get<2>(t) != 0 || std::get<3>(t) != 0)
					{
						auto k = std::get<0>(t);
						auto l = std::get<1>(t);

						while (k < source.height() && l < source.width())
						{
							if (source.mImage.at<float>(k, l) == 255.0f)
							{
								for (auto& candidate : candidates)
								{
									result.mImage.at<float>(std::get<0>(candidate), std::get<1>(candidate)) = 255.0f;
								}

								break;
							}

							candidates.push_back(std::make_tuple(std::get<0>(t), std::get<1>(t)));

							t = std::make_tuple(k + k - std::get<2>(t), l + l - std::get<3>(t), k, l);

							k = std::get<0>(t);
							l = std::get<1>(t);
						}
					}
					candidates.clear();
				}

				 t = std::make_tuple(0, 0, 0, 0);
				// bottom right
				if (source.mImage.at<float>(i + 1, j + 1) == 255.0f)
				{
					const float s = norme.mImage.at<float>(i, j) + norme.mImage.at<float>(i + 1, j + 1);
					const double sum1 = s + norme.mImage.at<float>(i + 2, j + 1); // bottom right right
					const double sum2 = s + norme.mImage.at<float>(i + 2, j + 2); // bottom right bottom right
					const double sum3 = s + norme.mImage.at<float>(i + 1, j + 2); // bottom right bottom
					if (sum1 > sum2)
					{
						if (sum1 > sum3)
						{
							if (sum1 > seuil)
								t = (std::make_tuple(i + 2, j + 1, i + 1, j + 1));
						}
						else
							if (sum3 > seuil)
								t = (std::make_tuple(i + 1, j + 2, i + 1, j + 1));
					}
					else
					{
						if (sum2 > sum3)
						{
							if (sum2 > seuil)
								t = (std::make_tuple(i + 2, j + 2, i + 1, j + 1));
						}
						else
							if (sum3 > seuil)
								t = (std::make_tuple(i + 1, j + 2, i + 1, j + 1));
					}
					if (std::get<0>(t) != 0 || std::get<1>(t) != 0 || std::get<2>(t) != 0 || std::get<3>(t) != 0)
					{
						auto k = std::get<0>(t);
						auto l = std::get<1>(t);

						while (k < source.height() && l < source.width())
						{
							if (source.mImage.at<float>(k, l) == 255.0f)
							{
								for (auto& candidate : candidates)
								{
									result.mImage.at<float>(std::get<0>(candidate), std::get<1>(candidate)) = 255.0f;
								}

								break;
							}

							candidates.push_back(std::make_tuple(std::get<0>(t), std::get<1>(t)));

							t = std::make_tuple(k + k - std::get<2>(t), l + l - std::get<3>(t), k, l);

							k = std::get<0>(t);
							l = std::get<1>(t);
						}
					}
					candidates.clear();
				}

				t = std::make_tuple(0, 0, 0, 0);
				// bottom
				if (source.mImage.at<float>(i, j + 1) == 255.0f)
				{
					const float s = norme.mImage.at<float>(i, j) + norme.mImage.at<float>(i, j + 1);
					const double sum1 = s + norme.mImage.at<float>(i + 1, j + 2); // bottom bottom right
					const double sum2 = s + norme.mImage.at<float>(i, j + 2); // bottom bottom
					const double sum3 = s + norme.mImage.at<float>(i - 1, j + 2); // bottom bottom left
					if (sum1 > sum2)
					{
						if (sum1 > sum3)
						{
							if (sum1 > seuil)
								t = (std::make_tuple(i + 1, j + 2, i, j + 1));
						}
						else
							if (sum3 > seuil)
								t = (std::make_tuple(i - 1, j + 2, i, j + 1));
					}
					else
					{
						if (sum2 > sum3)
						{
							if (sum2 > seuil)
								t = (std::make_tuple(i, j + 2, i, j + 1));
						}
						else
							if (sum3 > seuil)
								t = (std::make_tuple(i - 1, j + 2, i, j + 1));
					}
					if (std::get<0>(t) != 0 || std::get<1>(t) != 0 || std::get<2>(t) != 0 || std::get<3>(t) != 0)
					{
						auto k = std::get<0>(t);
						auto l = std::get<1>(t);

						while (k < source.height() && l < source.width())
						{
							if (source.mImage.at<float>(k, l) == 255.0f)
							{
								for (auto& candidate : candidates)
								{
									result.mImage.at<float>(std::get<0>(candidate), std::get<1>(candidate)) = 255.0f;
								}

								break;
							}

							candidates.push_back(std::make_tuple(std::get<0>(t), std::get<1>(t)));

							t = std::make_tuple(k + k - std::get<2>(t), l + l - std::get<3>(t), k, l);

							k = std::get<0>(t);
							l = std::get<1>(t);
						}
					}
					candidates.clear();
				}

				t = std::make_tuple(0, 0, 0, 0);
				// bottom left
				if (source.mImage.at<float>(i - 1, j + 1) == 255.0f)
				{
					const float s = norme.mImage.at<float>(i, j) + norme.mImage.at<float>(i - 1, j + 1);
					const double sum1 = s + norme.mImage.at<float>(i - 1, j + 2); // bottom left bottom
					const double sum2 = s + norme.mImage.at<float>(i - 2, j + 2); // bottom left bottom left
					const double sum3 = s + norme.mImage.at<float>(i - 2, j + 1); // bottom left left
					if (sum1 > sum2)
					{
						if (sum1 > sum3)
						{
							if (sum1 > seuil)
								t = (std::make_tuple(i - 1, j + 2, i - 1, j + 1));
						}
						else
							if (sum3 > seuil)
								t = (std::make_tuple(i - 2, j + 1, i - 1, j + 1));
					}
					else
					{
						if (sum2 > sum3)
						{
							if (sum2 > seuil)
								t = (std::make_tuple(i - 2, j + 2, i - 1, j + 1));
						}
						else
							if (sum3 > seuil)
								t = (std::make_tuple(i - 2, j + 1, i - 1, j + 1));
					}
					if (std::get<0>(t) != 0 || std::get<1>(t) != 0 || std::get<2>(t) != 0 || std::get<3>(t) != 0)
					{
						auto k = std::get<0>(t);
						auto l = std::get<1>(t);

						while (k < source.height() && l < source.width())
						{
							if (source.mImage.at<float>(k, l) == 255.0f)
							{
								for (auto& candidate : candidates)
								{
									result.mImage.at<float>(std::get<0>(candidate), std::get<1>(candidate)) = 255.0f;
								}

								break;
							}

							candidates.push_back(std::make_tuple(std::get<0>(t), std::get<1>(t)));

							t = std::make_tuple(k + k - std::get<2>(t), l + l - std::get<3>(t), k, l);

							k = std::get<0>(t);
							l = std::get<1>(t);
						}
					}
					candidates.clear();
				}

				t = std::make_tuple(0, 0, 0, 0);
				// left
				if (source.mImage.at<float>(i - 1, j) == 255.0f)
				{
					const float s = norme.mImage.at<float>(i, j) + norme.mImage.at<float>(i - 1, j);
					const double sum1 = s + norme.mImage.at<float>(i - 2, j + 1); // left bottom left
					const double sum2 = s + norme.mImage.at<float>(i - 2, j); // left left
					const double sum3 = s + norme.mImage.at<float>(i - 2, j - 1); // left top left 
					if (sum1 > sum2)
					{
						if (sum1 > sum3)
						{
							if (sum1 > seuil)
								t = (std::make_tuple(i - 2, j + 1, i - 1, j));
						}
						else
							if (sum3 > seuil)
								t = (std::make_tuple(i - 2, j - 1, i - 1, j));
					}
					else
					{
						if (sum2 > sum3)
						{
							if (sum2 > seuil)
								t = (std::make_tuple(i - 2, j, i - 1, j));
						}
						else
							if (sum3 > seuil)
								t = (std::make_tuple(i - 2, j - 1, i - 1, j));
					}
					if (std::get<0>(t) != 0 || std::get<1>(t) != 0 || std::get<2>(t) != 0 || std::get<3>(t) != 0)
					{
						auto k = std::get<0>(t);
						auto l = std::get<1>(t);

						while (k < source.height() && l < source.width())
						{
							if (source.mImage.at<float>(k, l) == 255.0f)
							{
								for (auto& candidate : candidates)
								{
									result.mImage.at<float>(std::get<0>(candidate), std::get<1>(candidate)) = 255.0f;
								}

								break;
							}

							candidates.push_back(std::make_tuple(std::get<0>(t), std::get<1>(t)));

							t = std::make_tuple(k + k - std::get<2>(t), l + l - std::get<3>(t), k, l);

							k = std::get<0>(t);
							l = std::get<1>(t);
						}
					}
					candidates.clear();
				}

				t = std::make_tuple(0, 0, 0, 0);
				// top left
				if (source.mImage.at<float>(i - 1, j - 1) == 255.0f)
				{
					const float s = norme.mImage.at<float>(i, j) + norme.mImage.at<float>(i - 1, j - 1);
					const double sum1 = s + norme.mImage.at<float>(i - 2, j - 1); // top left left
					const double sum2 = s + norme.mImage.at<float>(i - 2, j - 2); // top left top left
					const double sum3 = s + norme.mImage.at<float>(i - 1, j - 2); // top left top 
					if (sum1 > sum2)
					{
						if (sum1 > sum3)
						{
							if (sum1 > seuil)
								t = (std::make_tuple(i - 2, j - 1, i - 1, j - 1));
						}
						else
							if (sum3 > seuil)
								t = (std::make_tuple(i - 1, j - 2, i - 1, j - 1));
					}
					else
					{
						if (sum2 > sum3)
						{
							if (sum2 > seuil)
								t = (std::make_tuple(i - 2, j - 2, i - 1, j - 1));
						}
						else
							if (sum3 > seuil)
								t = (std::make_tuple(i - 1, j - 2, i - 1, j - 1));
					}
				}
				if (std::get<0>(t) != 0 || std::get<1>(t) != 0 || std::get<2>(t) != 0 || std::get<3>(t) != 0)
				{
					auto k = std::get<0>(t);
					auto l = std::get<1>(t);

					while (k < source.height() && l < source.width())
					{
						if (source.mImage.at<float>(k, l) == 255.0f)
						{
							for (auto& candidate : candidates)
							{
								result.mImage.at<float>(std::get<0>(candidate), std::get<1>(candidate)) = 255.0f;
							}

							break;
						}

						candidates.push_back(std::make_tuple(std::get<0>(t), std::get<1>(t)));

						t = std::make_tuple(k + k - std::get<2>(t), l + l - std::get<3>(t), k, l);

						k = std::get<0>(t);
						l = std::get<1>(t);
					}
				}
				candidates.clear();
			}

		}

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
