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


const int seuil_slider_max = 100;
int seuil_slider = 20;
float seuil;

const int seuil_hyst_slider_max = 100;
int seuil_hyst_slider = 10;
float seuil_hyst;
cv::Mat make_canvas(const std::vector<const cv::Mat*>& vecMat, int windowHeight, int nRows);


void seuillage (cv::Mat& source, cv::Mat& destination, float& threshold) {
    for (int j=0; j < source.size().width; ++j) {
        for (int i=0; i < source.size().height; ++i) {
            if (source.at<float>(i,j) < threshold*255)
                destination.at<float>(i,j) = 0;
            else
                destination.at<float>(i,j) = 255;
        }
    }
}


void seuillage_second (cv::Mat& source, cv::Mat& intermediaire, cv::Mat& destination, float& threshold) {
	for (int j=0; j < destination.size().width; ++j) {
        for (int i=0; i < destination.size().height; ++i) {
            if (source.at<float>(i,j) < threshold*255)
                destination.at<float>(i,j) = 0;
            else {
                int count = 0;
                //faire un test sur le max(0,convolutionsize())
                for (int k = -1; k < 2; ++k) {
                    for (int l = -1; l < 2; ++l) {
                        if (intermediaire.at<float>(i+k,j+l) == 255)
                            count++;
                    }
                }
                if (count != 0)
                    destination.at<float>(i,j) = 254;
            }
        }
    }
}

void seuil_hysteresis (cv::Mat& source, cv::Mat& intermediaire, cv::Mat& destination, float& threshold_haut, float& threshold_bas ) {
    /*seuillage(destinationNorme, seuilHaut, threshold_haut);
    seuillage_second (destinationNorme, seuilHaut, seuilHyst, threshold_bas);
    cv::imshow("haut", seuilHaut);*/
}

void on_trackbar( int, void* ) {
   /* seuil = float(seuil_slider)/seuil_slider_max;
    seuillage(destinationNorme, destinationNormeSeuil, seuil);
    cv::imshow("gradientNormeSeuil", destinationNormeSeuil);
    
    seuil_hyst = float(seuil_hyst_slider)/seuil_hyst_slider_max;
    seuil_hysteresis(destinationNorme, seuilHaut, seuilHyst, seuil, seuil_hyst);
    cv::imshow("Hysteresis", seuilHyst);*/
}

float findThreshold(const cv::Mat img) {
    
    return 1.f;
}

void bidirectionnal()
{
	
}

int main(int argc, const char * argv[]) {
	seuil = 0.0f;

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

	std::string threshold_type = json["seuillage"];

	Image destinationNorme, destinationDirection, destinationNormeGris, GrisChan[3];

	const unsigned new_height = image.height() - 2;
	const unsigned new_width = image.width() - 2;

	// bidirectionnal
    Image destinationX = image.convolution(bidirectionnal_kernels[0]);
    Image destinationY = image.convolution(bidirectionnal_kernels[1]);
	auto gradient = Image::bidirectionalGradient(destinationX, destinationY);
    
	// multidirectionnal
    Image destMulti0 = image.convolution(multidirectionnal_kernels[0]);
    Image destMulti1 = image.convolution(multidirectionnal_kernels[1]);
    Image destMulti2 = image.convolution(multidirectionnal_kernels[2]);
    Image destMulti3 = image.convolution(multidirectionnal_kernels[3]);
	
	Image grayMulti0 = destMulti0.toGray();
	const Image grayMulti1 = destMulti1.toGray();
	const Image grayMulti2 = destMulti2.toGray();
	const Image grayMulti3 = destMulti2.toGray();

	Image distMulti = Image::max(destMulti0.toGray(), destMulti1.toGray());
    distMulti = Image::max(distMulti, destMulti2.toGray());
    distMulti = Image::max(distMulti, destMulti3.toGray());
    distMulti.convertToFloat();
    
	auto dir_color = Image::multidirectionalDirection(distMulti, grayMulti0, grayMulti1, grayMulti2, grayMulti3);
	Image dirMulti = std::move(dir_color.first);
	Image dirColorMulti = std::move(dir_color.second);
    
    //thresholding
    cv::Scalar meanMulti, stddevMulti;
    cv::meanStdDev(distMulti._Mat(), meanMulti, stddevMulti, cv::Mat());
    cv::Scalar globalThreshold = meanMulti + 1.5 * stddevMulti;
    cv::Scalar highThreshold = meanMulti + 1.5 * stddevMulti;
    cv::Scalar lowThreshold = meanMulti + 1.2 * stddevMulti;
    
    Image globalThresholdingMulti = Image::thresholding(distMulti, globalThreshold.val[0]);
    Image HystHighThresholdingMulti = Image::thresholding(distMulti, highThreshold.val[0]);
    Image HystFinalThresholdingMulti = Image::thresholdingHysteresis(distMulti, highThreshold.val[0], lowThreshold.val[0]);
	
    seuil = 0.09f;
    seuil_hyst = 0.02f;
    
	/*
    seuillage(destinationNorme, destinationNormeSeuil, seuil);
    
    GrisChan[0] = destinationNorme/255;
    GrisChan[1] = destinationNorme/255;
    GrisChan[2] = destinationNorme/255;
    merge(GrisChan, 3, destinationNormeGris);
    
    /// Create Window
    cv::namedWindow("gradientNormeSeuil", 1);
    cv::namedWindow("Hysteresis", 1);
    
    /// Create Trackbars
    char TrackbarThresholdName[50];
    //sprintf( TrackbarThresholdName, "seuil");
    char TrackbarHystName[50];
    //sprintf( TrackbarHystName, "hysteresis");
    
    cv::createTrackbar( TrackbarThresholdName, "gradientNormeSeuil", &seuil_slider, seuil_slider_max, on_trackbar );
    cv::createTrackbar( TrackbarHystName, "Hysteresis", &seuil_hyst_slider, seuil_hyst_slider_max, on_trackbar );
    
    seuil_hysteresis (destinationNorme, seuilHaut, seuilHyst, seuil, seuil_hyst);
    /// Show some stuff
    on_trackbar( seuil_slider, 0 );
    on_trackbar( seuil_hyst_slider, 0 );*/
    

	const std::vector<const cv::Mat*> image_matrices = {
		&image._Mat(),
		&destinationX._Mat(),
		&destinationY._Mat(),
		&destMulti0._Mat(),
		&destMulti1._Mat(),
		&destMulti2._Mat(),
		&destMulti3._Mat(),
		&grayMulti0._Mat(),
		&distMulti._Mat(),
		&dirMulti._Mat(),
		&dirColorMulti._Mat(),
		&gradient.first._Mat(),
		&gradient.second._Mat(),
//        &globalThresholdingMulti._Mat(),
        &HystHighThresholdingMulti._Mat(),
        &HystFinalThresholdingMulti._Mat(),
		/*&destinationNormeGris._Mat(),
		&destinationDirection._Mat()*/

	};
	cv::imshow("AnalyseImage_TP1",make_canvas(image_matrices, 800, 4));
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