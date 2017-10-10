#pragma once
#include <opencv2/core/mat.hpp>

class Kernel;

class Image
{
public:
	Image();
	Image(unsigned height, unsigned width);
	Image(const Image& other)
		: mImage(other.mImage)
	{
	}

	Image(Image&& other) noexcept
		: mImage(std::move(other.mImage))
	{
	}

	Image& operator=(const Image& other)
	{
		if (this == &other)
			return *this;
		mImage = other.mImage;
		return *this;
	}

	Image& operator=(Image&& other) noexcept
	{
		if (this == &other)
			return *this;
		mImage = std::move(other.mImage);
		return *this;
	}

	bool readFromFile(const char* path);

	Image convolution(const Kernel& k) const;

	static std::pair<Image, Image> bidirectionalGradient(const Image& i1, const Image& i2);
	const cv::Mat& _Mat() const;

	unsigned width() const;
	unsigned height() const;
private:
	cv::Mat mImage;
};
