//
//  main.cpp
//  AnalyseImage-TP1
//
//  Created by Crymsius on 18/09/2017.
//  Copyright © 2017 Crymsius. All rights reserved.
//

#include <iostream>
#include <vector>
#include <tgmath.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

int convolution (cv::Mat image, cv::Mat derivee, int i, int j, int color) {
    // Problème sur les bords de l'image
    int temp = 0;
    for (int u = 0; u <= 2; u++) {
        for (int v = 0; v <= 2; v++) {
            temp = temp + (derivee.at<int>(u,v) * image.at<cv::Vec3b>(i+(u-1),j+(v-1))[color]);
        }
    }
    if (temp > 255) {
        return 255;
    } else if (temp < 0)
        return 0;
    else
        return temp;
}
float CalculMoyenne(unsigned a, unsigned b, unsigned c) {
    return float((a+b+c))/3.f;
}

int main(int argc, const char * argv[]) {
    cv::Mat image, destinationX, destinationY, destinationNorme, destinationDirection, destinationNormeSeuil;
    cv::Point anchor;
    double delta;
    int ddepth;
    cv::Mat moyenne;
    cv::Mat deriveeX;
    cv::Mat deriveeY;
    
    float seuil = 0.1f;
    
    moyenne = cv::Mat::ones(3,3,CV_8UC1);
    deriveeX = (cv::Mat_<int>(3,3) << 0, 0, 0, -1, 0, 1, 0, 0, 0);
    deriveeY = (cv::Mat_<int>(3,3) << 0, -1, 0, 0, 0, 0, 0, 1, 0);
    
    image = cv::imread("/Users/crymsius/Desktop/Boulot/Master/ID3D/Analyse et Traitement/TP/AnalyseImage-TP1/data/Lenna.png");
    destinationX = cv::Mat(image.size(),image.type());
    destinationY = cv::Mat(image.size(),image.type());
    destinationNorme = cv::Mat(image.size(), CV_32F);
    destinationNormeSeuil = cv::Mat(image.size(), CV_32F);
    destinationDirection = cv::Mat(image.size(), CV_32F);
    
    
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
    
    for (int j=1; j<image.size().height-1; ++j) {
        for (int i=1; i<image.size().width-1; ++i) {
            //On est dans l'image
            destinationX.at<cv::Vec3b>(i,j)[0] = convolution(image, deriveeX, i, j, 0);
            destinationX.at<cv::Vec3b>(i,j)[1] = convolution(image, deriveeX, i, j, 1);
            destinationX.at<cv::Vec3b>(i,j)[2] = convolution(image, deriveeX, i, j, 2);
            
            destinationY.at<cv::Vec3b>(i,j)[0] = convolution(image, deriveeY, i, j, 0);
            destinationY.at<cv::Vec3b>(i,j)[1] = convolution(image, deriveeY, i, j, 1);
            destinationY.at<cv::Vec3b>(i,j)[2] = convolution(image, deriveeY, i, j, 2);
            
            destinationNorme.at<float>(i,j) = sqrt(std::pow( CalculMoyenne (destinationX.at<cv::Vec3b>(i,j)[0],
                                                                             destinationX.at<cv::Vec3b>(i,j)[1],
                                                                             destinationX.at<cv::Vec3b>(i,j)[2]
                                                                             ),
                                                                2) +
                                                       std::pow( CalculMoyenne (destinationY.at<cv::Vec3b>(i,j)[0],
                                                                                destinationY.at<cv::Vec3b>(i,j)[1],
                                                                                destinationY.at<cv::Vec3b>(i,j)[2]
                                                                                ), 2)
                                                      );
            destinationDirection.at<float>(i,j) = cvFastArctan(CalculMoyenne (destinationY.at<cv::Vec3b>(i,j)[0],
                                                                                  destinationY.at<cv::Vec3b>(i,j)[1],
                                                                                  destinationY.at<cv::Vec3b>(i,j)[2]
                                                                                  ),
                                                                   CalculMoyenne (destinationX.at<cv::Vec3b>(i,j)[0],
                                                                                  destinationX.at<cv::Vec3b>(i,j)[1],
                                                                                  destinationX.at<cv::Vec3b>(i,j)[2]
                                                                                  )
                                                                   );
            if (destinationNorme.at<float>(i,j) < seuil*255)
                destinationNormeSeuil.at<float>(i,j) = 0;
            else
                destinationNormeSeuil.at<float>(i,j) = 255;
        }
    }
    cv::imshow("image", image);
//    cv::imshow("gradientX", destinationX);
//    cv::imshow("gradientY", destinationY);
    cv::imshow("gradientNorme", destinationNorme);
//    cv::imshow("gradientDirection", destinationDirection);
    cv::imshow("gradientNormeSeuil", destinationNormeSeuil);
    
    cv::waitKey(0);
    
    return 0;
}


