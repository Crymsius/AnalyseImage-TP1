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


Image Image::closure(const Image& contours, const Image& direction)
{

	const float seuil = +255.f + 255.f ;
	Image result;
	result.mImage = contours.mImage.clone();

	std::queue < std::tuple<unsigned, unsigned, unsigned , unsigned>> closure_candidates; // i, j, position du point précédent (pour éviter de boucler)


	// gérer en dehors les contours de l'image
	for (auto i = 2; i < contours.height() - 2; ++i)
	{
		for (auto j = 2; j < contours.width() - 2; ++j)
		{
			if (contours.mImage.at<float>(i, j) == 255.0f)
			{
				// top
				if (contours.mImage.at<float>(i, j - 1) == 255.0f)
				{
					const float s = direction.mImage.at<float>(i, j) + direction.mImage.at<float>(i, j - 1);
					const double sum1 = s + direction.mImage.at<float>(i - 1, j - 2); // top top left
					const double sum2 = s + direction.mImage.at<float>(i, j - 2); // top top
					const double sum3 = s + direction.mImage.at<float>(i + 1, j - 2); // top top right
					
					if (sum1 > sum2) 
					{
						if (sum1 > sum3) 
						{
							if (sum1 > seuil)
								closure_candidates.push(std::make_tuple(i - 1, j - 2, i, j - 1));
						}
						else if (sum3 > seuil)
								closure_candidates.push(std::make_tuple(i + 1, j - 2, i, j - 1));
					}
					else
					{
						if (sum2 > sum3)
						{
							if (sum2 > seuil)
								closure_candidates.push(std::make_tuple(i, j - 2, i, j - 1));
						}
						else if (sum3 > seuil)
								closure_candidates.push(std::make_tuple(i + 1, j - 2, i, j - 1));
					}
				}

				// top right
				if (contours.mImage.at<float>(i + 1, j - 1) == 255.0f)
				{
					const float s = direction.mImage.at<float>(i, j) + direction.mImage.at<float>(i + 1, j - 1);
					const double sum1 = s + direction.mImage.at<float>(i + 1, j - 2); // top right top
					const double sum2 = s + direction.mImage.at<float>(i + 1, j - 2); // top right top right
					const double sum3 = s + direction.mImage.at<float>(i + 2, j - 1); // top right right

					if (sum1 > sum2)
					{
						if (sum1 > sum3)
						{
							if (sum1 > seuil)
								closure_candidates.push(std::make_tuple(i + 1, j - 2, i + 1, j - 1));
						}
						else if (sum3 > seuil)
								closure_candidates.push(std::make_tuple(i + 2, j - 1, i + 1, j - 1));
					}
					else
					{
						if (sum2 > sum3)
						{
							if (sum2 > seuil)
								closure_candidates.push(std::make_tuple(i + 1, j - 2, i + 1, j - 1));
						}
						else if (sum3 > seuil)
							closure_candidates.push(std::make_tuple(i + 2, j - 1, i + 1, j - 1));
					}
				}

				// right
				if (contours.mImage.at<float>(i + 1, j) == 255.0f)
				{
					const float s = direction.mImage.at<float>(i, j) + direction.mImage.at<float>(i + 1, j);
					const double sum1 = s + direction.mImage.at<float>(i + 2, j - 1); // right top right
					const double sum2 = s + direction.mImage.at<float>(i + 2, j); // right right
					const double sum3 = s + direction.mImage.at<float>(i + 2, j + 1); // right bottom right
					if (sum1 > sum2)
					{
						if (sum1 > sum3)
						{
							if (sum1 > seuil)
								closure_candidates.push(std::make_tuple(i + 2, j - 1, i + 1, j));
						}
						else if (sum3 > seuil)
								closure_candidates.push(std::make_tuple(i + 2, j + 1, i + 1, j));
					}
					else if (sum2 > sum3)
					{
						if (sum2 > seuil)
							closure_candidates.push(std::make_tuple(i + 2, j, i + 1, j));
						else if (sum3 > seuil)
							closure_candidates.push(std::make_tuple(i + 2, j + 1, i + 1, j));
					}
				}

				// bottom right
				if (contours.mImage.at<float>(i + 1, j + 1) == 255.0f)
				{
					const float s = direction.mImage.at<float>(i, j) + direction.mImage.at<float>(i + 1, j + 1);
					const double sum1 = s + direction.mImage.at<float>(i + 2, j + 1); // bottom right right
					const double sum2 = s + direction.mImage.at<float>(i + 2, j + 2); // bottom right bottom right
					const double sum3 = s + direction.mImage.at<float>(i + 1, j + 2); // bottom right bottom
					if (sum1 > sum2)
					{
						if (sum1 > sum3)
						{
							if (sum1 > seuil)
								closure_candidates.push(std::make_tuple(i + 2, j + 1, i + 1, j + 1));
						}
						else
							if (sum3 > seuil)
								closure_candidates.push(std::make_tuple(i + 1, j + 2, i + 1, j + 1));
					}
					else
					{
						if (sum2 > sum3)
						{
							if (sum2 > seuil)
								closure_candidates.push(std::make_tuple(i + 2, j + 2, i + 1, j + 1));
						}
						else
							if (sum3 > seuil)
								closure_candidates.push(std::make_tuple(i + 1, j + 2, i + 1, j + 1));
					}
				}

				// bottom
				if (contours.mImage.at<float>(i, j + 1) == 255.0f)
				{
					const float s = direction.mImage.at<float>(i, j) + direction.mImage.at<float>(i, j + 1);
					const double sum1 = s + direction.mImage.at<float>(i + 1, j + 2); // bottom bottom right
					const double sum2 = s + direction.mImage.at<float>(i, j + 2); // bottom bottom
					const double sum3 = s + direction.mImage.at<float>(i - 1, j + 2); // bottom bottom left
					if (sum1 > sum2)
					{
						if (sum1 > sum3)
						{
							if (sum1 > seuil)
								closure_candidates.push(std::make_tuple(i + 1, j + 2, i, j + 1));
						}
						else
							if (sum3 > seuil)
								closure_candidates.push(std::make_tuple(i - 1, j + 2, i, j + 1));
					}
					else
					{
						if (sum2 > sum3)
						{
							if (sum2 > seuil)
								closure_candidates.push(std::make_tuple(i, j + 2, i, j + 1));
						}
						else
							if (sum3 > seuil)
								closure_candidates.push(std::make_tuple(i - 1, j + 2, i, j + 1));
					}
				}

				// bottom left
				if (contours.mImage.at<float>(i - 1, j + 1) == 255.0f)
				{
					const float s = direction.mImage.at<float>(i, j) + direction.mImage.at<float>(i - 1, j + 1);
					const double sum1 = s + direction.mImage.at<float>(i - 1, j + 2); // bottom left bottom
					const double sum2 = s + direction.mImage.at<float>(i - 2, j + 2); // bottom left bottom left
					const double sum3 = s + direction.mImage.at<float>(i - 2, j + 1); // bottom left left
					if (sum1 > sum2)
					{
						if (sum1 > sum3)
						{
							if (sum1 > seuil)
								closure_candidates.push(std::make_tuple(i - 1, j + 2, i - 1, j + 1));
						}
						else
							if (sum3 > seuil)
								closure_candidates.push(std::make_tuple(i - 2, j + 1, i - 1, j + 1));
					}
					else
					{
						if (sum2 > sum3)
						{
							if (sum2 > seuil)
								closure_candidates.push(std::make_tuple(i - 2, j + 2, i - 1, j + 1));
						}
						else
							if (sum3 > seuil)
								closure_candidates.push(std::make_tuple(i - 2, j + 1, i - 1, j + 1));
					}
				}

				// left
				if (contours.mImage.at<float>(i - 1, j) == 255.0f)
				{
					const float s = direction.mImage.at<float>(i, j) + direction.mImage.at<float>(i - 1, j);
					const double sum1 = s + direction.mImage.at<float>(i - 2, j + 1); // left bottom left
					const double sum2 = s + direction.mImage.at<float>(i - 2, j); // left left
					const double sum3 = s + direction.mImage.at<float>(i - 2, j - 1); // left top left 
					if (sum1 > sum2)
					{
						if (sum1 > sum3)
						{
							if (sum1 > seuil)
								closure_candidates.push(std::make_tuple(i - 2, j + 1, i - 1, j));
						}
						else
							if (sum3 > seuil)
								closure_candidates.push(std::make_tuple(i - 2, j - 1, i - 1, j));
					}
					else
					{
						if (sum2 > sum3)
						{
							if (sum2 > seuil)
								closure_candidates.push(std::make_tuple(i - 2, j, i - 1, j));
						}
						else
							if (sum3 > seuil)
								closure_candidates.push(std::make_tuple(i - 2, j - 1, i - 1, j));
					}
				}

				// top left
				if (contours.mImage.at<float>(i - 1, j - 1) == 255.0f)
				{
					const float s = direction.mImage.at<float>(i, j) + direction.mImage.at<float>(i - 1, j - 1);
					const double sum1 = s + direction.mImage.at<float>(i - 2, j - 1); // top left left
					const double sum2 = s + direction.mImage.at<float>(i - 2, j - 2); // top left top left
					const double sum3 = s + direction.mImage.at<float>(i - 1, j - 2); // top left top 
					if (sum1 > sum2)
					{
						if (sum1 > sum3)
						{
							if (sum1 > seuil)
								closure_candidates.push(std::make_tuple(i - 2, j - 1, i - 1, j - 1));
						}
						else
							if (sum3 > seuil)
								closure_candidates.push(std::make_tuple(i - 1, j - 2, i - 1, j - 1));
					}
					else
					{
						if (sum2 > sum3)
						{
							if (sum2 > seuil)
								closure_candidates.push(std::make_tuple(i - 2, j - 2, i - 1, j - 1));
						}
						else
							if (sum3 > seuil)
								closure_candidates.push(std::make_tuple(i - 1, j - 2, i - 1, j - 1));
					}
				}
			}

			std::vector<std::tuple<unsigned, unsigned>> already_candidates;
			already_candidates.reserve(100);

			while(!closure_candidates.empty())
			{
				auto tuple = closure_candidates.front();
				result.mImage.at<float>(std::get<0>(tuple), std::get<1>(tuple)) = 255.0f;
				closure_candidates.pop();

				if (std::find(already_candidates.begin(), already_candidates.end(), std::make_tuple(std::get<0>(tuple), std::get<1>(tuple))) != already_candidates.end())
				{
					continue;
				}

				already_candidates.push_back(std::make_tuple(std::get<0>(tuple), std::get<1>(tuple)));
				const auto k = std::get<0>(tuple);
				const auto l = std::get<1>(tuple);

				auto t = std::make_tuple(k + k - std::get<2>(tuple), l + l - std::get<3>(tuple));

				if (std::get<0>(t) < result.height() && std::get<1>(t) < result.width() && result.mImage.at<float>(std::get<0>(t), std::get<1>(t)) == 0)
				{

					const float sum = direction.mImage.at<float>(k, l) + direction.mImage.at<float>(std::get<0>(t), std::get<1>(t));

					if (sum > seuil)
						closure_candidates.push(std::make_tuple(std::get<0>(t), std::get<1>(t), std::get<0>(tuple), std::get<1>(tuple)));
				}
				
			}
		}
	}
	return std::move(result);
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
