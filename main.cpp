//
//  main.cpp
//  AnalyseImage-TP1
//
//  Created by Crymsius on 18/09/2017.
//  Copyright © 2017 Crymsius. All rights reserved.
//

#include <iostream>
#include <vector>
#if defined(_WIN32) || defined(WIN32)
#include <ctgmath>
#define _USE_MATH_DEFINES
#define USE_MATH_DEFINES
#include <math.h>
#else
#include <tgmath.h>
#endif
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "Kernel.h"
#include "Image.h"
#include "json.hpp"
#include <fstream>

cv::Mat make_canvas(const std::vector<const cv::Mat*>& vecMat, int windowHeight, int nRows);

int main(int argc, const char * argv[]) {
	

	std::ifstream stream("config.json");
	nlohmann::json json;
	stream >> json;

	std::vector<Kernel> bidirectionnal_kernels(2);
	auto t = 0;
	if (json["bidirectionnal"].size() < 2)
	{
		std::cerr << "Not enough bidirectionnal kernel, expected 2" << std::endl;
		exit(-2);
	}

	for (auto elmt : json["bidirectionnal"])
	{
		if (t == 2)
			break;
		std::vector<std::vector<int>> tk(elmt.size());
		for (auto i = 0; i < tk.size(); ++i)
		{
			tk[i].reserve(elmt[i].size());
			for (auto j = 0; j < elmt[i].size(); ++j)
				tk[i].push_back(elmt[i][j]);
		}
		bidirectionnal_kernels[t] = std::move(Kernel(tk));
		++t;
	}

	std::vector<Kernel> multidirectionnal_kernels(4);
	t = 0;
	if (json["multidirectionnal"].size() < 4)
	{
		std::cerr << "Not enough multidirectionnal kernel, expected 4" << std::endl;
		exit(-3);
	}
	for (auto elmt : json["multidirectionnal"])
	{
		if (t == 4)
			break;
		std::vector<std::vector<int>> tk(elmt.size());
		for (auto i = 0; i < tk.size(); ++i)
		{
			tk[i].reserve(elmt[i].size());
			for (auto j = 0; j < elmt[i].size(); ++j)
				tk[i].push_back(elmt[i][j]);
		}
		multidirectionnal_kernels[t] = std::move(Kernel(tk));
		++t;
	}
	std::string path = json["image_path"];

	Image image;
	if (!image.readFromFile(path.c_str()))
	{
		std::cerr << "Image cannot be read" << std::endl;
		exit(-4);
	}

	Image destinationNorme, destinationDirection, destinationNormeGris, GrisChan[3];

	const unsigned new_height = image.height() - 2;
	const unsigned new_width = image.width() - 2;
    
	// bidirectionnal
    Image destinationX = image.convolution(bidirectionnal_kernels[0]);
    Image destinationY = image.convolution(bidirectionnal_kernels[1]);
	auto gradient = Image::bidirectionalGradient(destinationX, destinationY);
    
	// multidirectionnal
    Image convoMulti0 = image.convolution(multidirectionnal_kernels[0]);
    Image convoMulti1 = image.convolution(multidirectionnal_kernels[1]);
    Image convoMulti2 = image.convolution(multidirectionnal_kernels[2]);
    Image convoMulti3 = image.convolution(multidirectionnal_kernels[3]);
	
    Image grayMulti0 = convoMulti0.toGray();
    Image grayMulti1 = convoMulti1.toGray();
    Image grayMulti2 = convoMulti2.toGray();
    Image grayMulti3 = convoMulti2.toGray();
    grayMulti0.convertToFloat();
    grayMulti1.convertToFloat();
    grayMulti2.convertToFloat();
    grayMulti3.convertToFloat();

    Image distMulti = Image::max(convoMulti0, convoMulti1);
    distMulti = Image::max(distMulti, convoMulti2);
    distMulti = Image::max(distMulti, convoMulti3);
    
    Image distMultiGray = distMulti.toGray();
    distMultiGray.convertToFloat();
	
    auto dir_color = Image::multidirectionalDirection(distMultiGray, grayMulti0, grayMulti1, grayMulti2, grayMulti3);
//    auto dir_color = Image::multidirectionalDirection(distMulti, convoMulti0, convoMulti1, convoMulti2, convoMulti3);
	Image dirMulti = std::move(dir_color.first);
	Image dirColorMulti = std::move(dir_color.second);

    //thresholding
    cv::Scalar meanMulti, stddevMulti;
    cv::meanStdDev(distMultiGray._Mat(), meanMulti, stddevMulti, cv::Mat());
    cv::Scalar globalThreshold = meanMulti + 1.5 * stddevMulti;
    cv::Scalar highThreshold = meanMulti + 1.5 * stddevMulti;
    cv::Scalar lowThreshold = meanMulti + 1.2 * stddevMulti;

	
    Image globalThresholdingMulti = Image::thresholding(distMultiGray, globalThreshold.val[0]);
    Image localThresholdingMulti = Image::localThresholding(distMultiGray, 20);
    Image hystHighThresholdingMulti = Image::thresholding(distMultiGray, highThreshold.val[0]);
    Image hystFinalThresholdingMulti = Image::thresholdingHysteresis(distMultiGray, highThreshold.val[0], lowThreshold.val[0]);
	
    Image thinMulti = Image::thinningMulti(globalThresholdingMulti, gradient.first, gradient.second);
	    
	const std::vector<const cv::Mat*> image_matrices = {
		&image._Mat(),
		&destinationX._Mat(),
		&destinationY._Mat(),
		&convoMulti0._Mat(),
		&convoMulti1._Mat(),
		&convoMulti2._Mat(),
		&convoMulti3._Mat(),
        &distMultiGray._Mat(),
		&dirColorMulti._Mat(),
		&gradient.first._Mat(),
		&gradient.second._Mat()

	};
	cv::imshow("AnalyseImage_TP1",make_canvas(image_matrices, 800, 3));
    thinMulti.show("affinage");
    globalThresholdingMulti.show("globalThreshold");
    localThresholdingMulti.show("localThreshold");
	hystFinalThresholdingMulti.show("hystThreashold");
    cv::waitKey(0);
    
    return 0;
}


/**
* @brief makeCanvas Makes composite image from the given images
* @param vecMat Vector of Images.
* @param windowHeight The height of the new composite image to be formed.
* @param nRows Number of rows of images. (Number of columns will be calculated
*              depending on the value of total number of images).
* @return new composite image.
*/
cv::Mat make_canvas(const std::vector<const cv::Mat*>& vecMat, int windowHeight, int nRows) {
	int N = vecMat.size();
	nRows = nRows > N ? N : nRows;
	const int edgeThickness = 10;
	const int imagesPerRow = ceil(double(N) / nRows);
	const int resizeHeight = floor(2.0 * ((floor(double(windowHeight - edgeThickness) / nRows)) / 2.0)) - edgeThickness;
	int maxRowLength = 0;

	std::vector<int> resizeWidth;
	for (int i = 0; i < N;) {
		int thisRowLen = 0;
		for (int k = 0; k < imagesPerRow; k++) {
			double aspectRatio = double(vecMat[i]->cols) / vecMat[i]->rows;
			int temp = int(ceil(resizeHeight * aspectRatio));
			resizeWidth.push_back(temp);
			thisRowLen += temp;
			if (++i == N) break;
		}
		if ((thisRowLen + edgeThickness * (imagesPerRow + 1)) > maxRowLength) {
			maxRowLength = thisRowLen + edgeThickness * (imagesPerRow + 1);
		}
	}
	int windowWidth = maxRowLength;
	cv::Mat canvasImage(windowHeight, windowWidth, CV_8UC3, cv::Scalar(0, 0, 0));

	for (int k = 0, i = 0; i < nRows; i++) {
		int y = i * resizeHeight + (i + 1) * edgeThickness;
		int x_end = edgeThickness;
		for (int j = 0; j < imagesPerRow && k < N; k++, j++) {
			int x = x_end;
			cv::Rect roi(x, y, resizeWidth[k], resizeHeight);
			cv::Size s = canvasImage(roi).size();
			// change the number of channels to three
			cv::Mat target_ROI(s, CV_8UC3);
			if (vecMat[k]->channels() != canvasImage.channels()) {
				if (vecMat[k]->channels() == 1) {
					cv::cvtColor(*vecMat[k], target_ROI, CV_GRAY2BGR);
				}
			}
			else {
				vecMat[k]->copyTo(target_ROI);
			}
			cv::resize(target_ROI, target_ROI, s);
			if (target_ROI.type() != canvasImage.type()) {
				target_ROI.convertTo(target_ROI, canvasImage.type());
			}
			target_ROI.copyTo(canvasImage(roi));
			x_end += resizeWidth[k] + edgeThickness;
		}
	}
	return canvasImage;
}