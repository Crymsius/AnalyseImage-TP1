//
//  main.cpp
//  AnalyseImage-TP1
//
//  Created by Crymsius on 18/09/2017.
//  Copyright © 2017 Crymsius. All rights reserved.
//

#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

int convolution (cv::Mat image, cv::Mat derivee, int i, int j, int color) {
    // Cas sur les côtés où image n'est pas définie -> mettre des 0
    int temp = 0;
    for (int u = 0; u <= 2; u++) {
        for (int v = 0; v <= 2; v++) {
            temp = temp + (derivee.at<int>(u,v) * image.at<cv::Vec3b>(i+(u-1),j+(v-1))[color]);
         //   std::cout<< temp << "/";
        }
    }
//    std::cout << "\n";
    if (temp > 255) {
//      std::cout << "error";
        return 255;
    } else if (temp < 0)
        return 0;
    else
        return temp;
}

int main(int argc, const char * argv[]) {
    cv::Mat image, destinationX, destinationY;
    cv::Point anchor;
    double delta;
    int ddepth;
    cv::Mat moyenne;
    cv::Mat deriveeX;
    cv::Mat deriveeY;
    
    moyenne = cv::Mat::ones(3,3,CV_8UC1);
    deriveeX = (cv::Mat_<int>(3,3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);
    deriveeY = (cv::Mat_<int>(3,3) << -1, -1, -1, 0, 0, 0, 1, 1, 1);
    
    //std::cout << "deriveeX = " << std::endl << " " << deriveeX << std::endl << std::endl;
    //std::cout << "deriveeY = " << std::endl << " " << deriveeY << std::endl << std::endl;
    //std::cout << deriveeX.at<int>(2,2);
    image = cv::imread("/Users/crymsius/Desktop/Boulot/Master/ID3D/Analyse et Traitement/TP/AnalyseImage-TP1/data/Lenna.png");
    destinationX = cv::imread("/Users/crymsius/Desktop/Boulot/Master/ID3D/Analyse et Traitement/TP/AnalyseImage-TP1/data/Lenna.png");
    destinationY = cv::imread("/Users/crymsius/Desktop/Boulot/Master/ID3D/Analyse et Traitement/TP/AnalyseImage-TP1/data/Lenna.png");
    if (!image.data) {
        printf("erreur image");
        return -1;
    }
    cv::namedWindow("image", 1);
    
    /// Initialize arguments for the filter
    anchor = cv::Point(-1, -1);
    delta = 0;
    ddepth = -1;
    
    //filter2D(image, destination, ddepth, deriveeY, anchor, delta, cv::BORDER_DEFAULT );
    
    for (int i=1; i<image.size().height-1; i++) {
        for (int j=1; j<image.size().width-1; j++) {
            //On est dans l'image
            //std::cout << "destination = " << std::endl << " " << destination.at<cv::Vec3b>(1,1) << std::endl << std::endl;
            destinationX.at<cv::Vec3b>(i,j)[0] = convolution(image, deriveeX, i, j, 0);
            destinationX.at<cv::Vec3b>(i,j)[1] = convolution(image, deriveeX, i, j, 1);
            destinationX.at<cv::Vec3b>(i,j)[2] = convolution(image, deriveeX, i, j, 2);
            
            destinationY.at<cv::Vec3b>(i,j)[0] = convolution(image, deriveeY, i, j, 0);
            destinationY.at<cv::Vec3b>(i,j)[1] = convolution(image, deriveeY, i, j, 1);
            destinationY.at<cv::Vec3b>(i,j)[2] = convolution(image, deriveeY, i, j, 2);

        }
    }
    cv::imshow("image", image);
    cv::imshow("gradientX", destinationX);
    cv::imshow("gradientY", destinationY);
    cv::waitKey(0);
    
    return 0;
}


