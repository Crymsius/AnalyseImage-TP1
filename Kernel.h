#pragma once
#include <vector>
#include <opencv2/core/mat.hpp>

class Kernel
{
public:
	explicit Kernel(const std::vector<std::vector<int>>& mat);

	Kernel(const Kernel& other)
		: mKernel(other.mKernel)
	{
	}

	Kernel(Kernel&& other) noexcept
		: mKernel(std::move(other.mKernel))
	{
	}

	Kernel& operator=(const Kernel& other)
	{
		if (this == &other)
			return *this;
		mKernel = other.mKernel;
		return *this;
	}

	Kernel& operator=(Kernel&& other) noexcept
	{
		if (this == &other)
			return *this;
		mKernel = std::move(other.mKernel);
		return *this;
	}

	const cv::Mat& _Mat() const;

private:
	cv::Mat mKernel;
};
