#pragma once
#include <opencv2/core/mat.hpp>

class Kernel;

class Image
{
public:
	Image();
	Image(unsigned height, unsigned width);
	Image(const Image& other)
		: mImage(other.mImage.clone())
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
		mImage = other.mImage.clone();
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
	Image toGray() const;
    void convertToFloat ();
	cv::Mat* _MatPtr();

	static std::pair<Image, Image> bidirectionalGradient(const Image& i1, const Image& i2);
	static std::pair<Image, Image> multidirectionalDirection(const Image& distance, const Image& gray0, const Image& gray1,
	                                                         const Image& gray2, const Image& gray3);
    
    static float calcLocalThreshold(const Image& source, int size, int i, int j);
    static Image localThresholding(const Image& source, const int size);
    static Image thresholding(const Image& source, const float& threshold);
    static Image thresholdingHysteresis(const Image& source, const float& thresholdHigh, const float& thresholdLow);
    static Image thresholdingLow(const Image& source, const Image& temp, const float& threshold);
    
    static Image thinningMulti(const Image& source, const Image& grad, const Image& dir);
    
    static Image max(const Image& i0, const Image& i1);
	static Image closure(const Image& source, const Image& norme);

	const cv::Mat& _Mat() const;

	const float& operator()(unsigned i, unsigned j) const;

	void show(const char* name) const;

	unsigned width() const;
	unsigned height() const;
private:
	cv::Mat mImage;
	enum class Direction
	{
		Left,
		TopLeft,
		Top,
		TopRight,
		Right,
		BottomRight,
		Bottom,
		BottomLeft
	};
};
