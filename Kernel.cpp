#include "Kernel.h"

Kernel::Kernel(const std::vector<std::vector<int>>& mat)
	: mKernel(mat.size(), mat[0].size(), CV_32SC1)
{
	for (auto col = 0; col < mat.size(); ++col)
	{
		for (auto row = 0; row < mat[col].size(); ++row)
		{
			mKernel.at<int>(col, row) = mat[col][row];
		}
	}
}

const cv::Mat& Kernel::_Mat() const
{
	return mKernel;
}

