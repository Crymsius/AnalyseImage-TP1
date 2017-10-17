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

cv::Mat image, destinationX, destinationY, destinationNorme, destinationDirection, destinationNormeSeuil, destinationNormeGris, GrisChan[3], seuilHaut, seuilHyst;
cv::Mat destMulti0, destMulti1, destMulti2, destMulti3, distMulti, distMultiSeuil, dirMulti, dirColorMulti;
cv::Mat grayMulti0, grayMulti1, grayMulti2, grayMulti3;
cv::Mat MultiThin;
const int seuil_slider_max = 100;
int seuil_slider = 100;
float seuil;

const int seuil_hyst_slider_max = 100;
int seuil_hyst_slider = 10;
float seuil_hyst;

void seuillage (cv::Mat& source, cv::Mat& destination, float& threshold) {
    for (int i=0; i < source.size().height; ++i) {
        for (int j=0; j < source.size().width; ++j) {
            if (source.at<float>(i,j) < threshold*255) {
                destination.at<float>(i,j) = 0;
//                destination.at<uchar>(i,j) = 0;
            } else {
                destination.at<float>(i,j) = 255;
//                destination.at<uchar>(i,j) = 255;
            }
        }
    }
}


void seuillage_second (cv::Mat& source, cv::Mat& intermediaire, cv::Mat& destination, float& threshold) {
    int count;
    for (int j=0; j < destination.size().width; ++j) {
        for (int i=0; i < destination.size().height; ++i) {
            if (source.at<float>(i,j) < threshold*255)
                destination.at<float>(i,j) = 0;
            else {
                count = 0;
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
    seuillage(destinationNorme, seuilHaut, threshold_haut);
    seuillage_second (destinationNorme, seuilHaut, seuilHyst, threshold_bas);
    cv::imshow("haut", seuilHaut);
}

void affinageMultiDir (cv::Mat& source, cv::Mat& destination, cv::Mat& gradient, cv::Mat& direction) {
    for (int i=1; i < gradient.size().height-1; ++i) {
        for (int j=1; j < gradient.size().width-1; ++j) {
            if (source.at<float>(i,j) == 255) {
                double angle = direction.at<float>(i,j) + M_PI;
                if (fmod(angle, M_PI) == 0) {
                    if (gradient.at<float>(i,j) >= gradient.at<float>(i,j-1) && gradient.at<float>(i,j) >= gradient.at<float>(i,j+1))
                        destination.at<float>(i,j) = 255;
                    else
                        destination.at<float>(i,j) = 0;
                } else if ( fmod(angle, M_PI) == M_PI_4) {
                    if (gradient.at<float>(i,j) >= gradient.at<float>(i-1,j-1) && gradient.at<float>(i,j) >= gradient.at<float>(i+1,j+1))
                        destination.at<float>(i,j) = 255;
                    else
                        destination.at<float>(i,j) = 0;
                } else if (fmod(angle, M_PI) == M_PI_2) {
                    if (gradient.at<float>(i,j) >= gradient.at<float>(i-1,j) && gradient.at<float>(i,j) >= gradient.at<float>(i+1,j))
                        destination.at<float>(i,j) = 255;
                    else
                        destination.at<float>(i,j) = 0;
                } else if ( fmod(angle, M_PI) == 3 * (M_PI_4)) {
                    if (gradient.at<float>(i,j) >= gradient.at<float>(i+1,j-1) && gradient.at<float>(i,j) >= gradient.at<float>(i-1,j+1))
                        destination.at<float>(i,j) = 255;
                    else
                        destination.at<float>(i,j) = 0;
                }
            } else {
                destination.at<float>(i,j) = 0;
            }
        }
    }
}

void affinageBiDir(cv::Mat& source, cv::Mat& destination, cv::Mat& gradient, cv::Mat& direction) {
    for (int i=1; i < gradient.size().height-1; ++i) {
        for (int j=1; j < gradient.size().width-1; ++j) {
            if (source.at<float>(i,j) == 255) {
                double angle = direction.at<float>(i,j) + 180;
                if (fmod(angle,180) < 22.5) {
                    if (gradient.at<float>(i,j) >= gradient.at<float>(i,j-1) && gradient.at<float>(i,j) >= gradient.at<float>(i,j+1))
                        destination.at<float>(i,j) = 255;
                    else
                        destination.at<float>(i,j) = 0;
                } else if (fmod(angle,180) < 67.5) {
                    if (gradient.at<float>(i,j) >= gradient.at<float>(i-1,j-1) && gradient.at<float>(i,j) >= gradient.at<float>(i+1,j+1))
                        destination.at<float>(i,j) = 255;
                    else
                        destination.at<float>(i,j) = 0;
                } else if (fmod(angle,180) < 112.5) {
                    if (gradient.at<float>(i,j) >= gradient.at<float>(i-1,j) && gradient.at<float>(i,j) >= gradient.at<float>(i+1,j))
                        destination.at<float>(i,j) = 255;
                    else
                        destination.at<float>(i,j) = 0;
                } else if (fmod(angle,180) < 157.5) {
                    if (gradient.at<float>(i,j) >= gradient.at<float>(i+1,j-1) && gradient.at<float>(i,j) >= gradient.at<float>(i-1,j+1))
                        destination.at<float>(i,j) = 255;
                    else
                        destination.at<float>(i,j) = 0;
                } else if (fmod(angle,180) < 180) {
                    if (gradient.at<float>(i,j) >= gradient.at<float>(i,j-1) && gradient.at<float>(i,j) >= gradient.at<float>(i,j+1))
                        destination.at<float>(i,j) = 255;
                    else
                        destination.at<float>(i,j) = 0;
                }
            } else {
                destination.at<float>(i,j) = 0;
            }
        }
    }
}

void on_trackbar( int, void* ) {
    seuil = (float) seuil_slider/seuil_slider_max;
//    seuillage(destinationNorme, destinationNormeSeuil, seuil);
    seuillage(distMulti, distMultiSeuil, seuil);

    cv::imshow("gradientNormeSeuil", distMultiSeuil);
    
    seuil_hyst = (float) seuil_hyst_slider/seuil_hyst_slider_max;
    seuil_hysteresis(destinationNorme, seuilHaut, seuilHyst, seuil, seuil_hyst);
    cv::imshow("Hysteresis", seuilHyst);
    
    affinageMultiDir(distMultiSeuil, MultiThin, distMulti,dirMulti);
    cv::imshow("affinage", MultiThin);
}

int convolution (cv::Mat image, cv::Mat derivee, int i, int j, int color) {
    // Problème sur les bords de l'image
    int temp = 0;
    for (int u = 0; u <= 2; u++) {
        for (int v = 0; v <= 2; v++) {
            temp = temp + (derivee.at<int>(u,v) * image.at<cv::Vec3b>(i+(u-1),j+(v-1))[color]);
        }
    }
    temp = abs(temp);
    if (temp > 255) {
        return 254;
    }
    else
        return temp;
}

cv::Mat imageConvolution (cv::Mat image, cv::Mat noyau) {
    cv::Mat imageConvoluee(image.size().height-2,image.size().width-2,image.type());
    
    for (int i=1; i < image.size().height-1; ++i) {
        for (int j=1; j < image.size().width-1; ++j) {
            //On est dans l'image
            for (int k = 0; k < image.channels(); k++) {
                imageConvoluee.at<cv::Vec3b>(i-1,j-1)[k] = convolution(image, noyau, i, j, k);
            }
        }
    }
    return imageConvoluee;
}

float CalculMoyenne(unsigned a, unsigned b, unsigned c) {
    return float((a+b+c))/3.f;
}

float findThreshold(const cv::Mat img) {
    
    return 1.f;
}

cv::Mat directionMulti(cv::Mat dist, cv::Mat& dirColor, cv::Mat a0, cv::Mat a1, cv::Mat a2, cv::Mat a3){
    cv::Mat result(dist.size(), CV_32F);
    
    for (int i=0; i < dist.size().height; ++i) {
        for (int j=0; j < dist.size().width; ++j) {
            if (dist.at<float>(i,j) == 0) {
                result.at<float>(i,j) = 0;
                dirColor.at<cv::Vec3b>(i,j) = cv::Vec3b(0,0,0);
            } else if (dist.at<float>(i,j) == a0.at<uchar>(i,j)) {
                result.at<float>(i,j) = 0;
                dirColor.at<cv::Vec3b>(i,j) = cv::Vec3b(200,0,0);
            } else if (dist.at<float>(i,j) == a1.at<uchar>(i,j)) {
                result.at<float>(i,j) = 1 * (M_PI_4);
                dirColor.at<cv::Vec3b>(i,j) = cv::Vec3b(0,200,0);
            } else if (dist.at<float>(i,j) == a2.at<uchar>(i,j)) {
                result.at<float>(i,j) = 2 * (M_PI_4);
                dirColor.at<cv::Vec3b>(i,j) = cv::Vec3b(0,0,200);
            } else {
                result.at<float>(i,j) = 3 * (M_PI_4);
                dirColor.at<cv::Vec3b>(i,j) = cv::Vec3b(255,255,255);
            }
        }
    }

    return result;
}

int main(int argc, const char * argv[]) {
    
    cv::Mat moyenne;
    cv::Mat deriveeX;
    cv::Mat deriveeY;
    cv::Mat multi0, multi1, multi2, multi3;
    
    seuil = 0.0f;
    
    moyenne = cv::Mat::ones(3,3,CV_8UC1);
    deriveeX = (cv::Mat_<int>(3,3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);
    deriveeY = (cv::Mat_<int>(3,3) << -1, -1, -1, 0, 0, 0, 1, 1, 1);
    
    multi0 = (cv::Mat_<int>(3,3) << -3, -3, 5, -3, 0, 5, -3, -3, 5);
    multi1 = (cv::Mat_<int>(3,3) << -3, 5, 5, -3, 0, 5, -3, -3, -3);
    multi2 = (cv::Mat_<int>(3,3) << 5, 5, 5, -3, 0, -3, -3, -3, -3);
    multi3 = (cv::Mat_<int>(3,3) << 5, 5, -3, 5, 0, -3, -3, -3, -3);
    
    image = cv::imread("data/Lenna.png");
//    image = cv::imread("data/1.png");
    destinationX = cv::Mat(image.size().height-2,image.size().width-2,image.type());
    destinationY = cv::Mat(image.size().height-2,image.size().width-2,image.type());
    
    destinationNorme = cv::Mat(image.size().height-2,image.size().width-2, CV_32F);
    destinationNormeGris = cv::Mat(image.size().height-2,image.size().width-2, image.type());
    
    destinationDirection = cv::Mat(image.size().height-2,image.size().width-2, CV_32F);

    destinationNormeSeuil = cv::Mat(image.size().height-2,image.size().width-2, CV_32F);
    seuilHaut = cv::Mat(image.size().height-2,image.size().width-2, CV_32F);
    seuilHyst = cv::Mat(seuilHaut.size().height-2,seuilHaut.size().width-2, CV_32F);
    
    destMulti0 = cv::Mat(image.size().height-2,image.size().width-2,image.type());
    destMulti1 = cv::Mat(image.size().height-2,image.size().width-2,image.type());
    destMulti2 = cv::Mat(image.size().height-2,image.size().width-2,image.type());
    destMulti3 = cv::Mat(image.size().height-2,image.size().width-2,image.type());
    
    distMulti = cv::Mat(destMulti0.size(), CV_32F);
    distMultiSeuil = cv::Mat(distMulti.size(), CV_32F);
    MultiThin = cv::Mat(distMulti.size().height-2, distMulti.size().width-2, CV_32F);
    dirColorMulti = cv::Mat(distMulti.size(), image.type());
    
    if (!image.data) {
        printf("erreur image");
        return -1;
    }
    cv::namedWindow("image", 1);
    
    //convolutions:
    destinationX = imageConvolution(image, deriveeX);
    destinationY = imageConvolution(image, deriveeY);
    
    for (int j=0; j < destinationX.size().width; ++j) {
        for (int i=0; i < destinationX.size().height; ++i) {
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
            destinationDirection.at<float>(i,j) = cvFastArctan( CalculMoyenne (destinationY.at<cv::Vec3b>(i,j)[0],
                                                                                  destinationY.at<cv::Vec3b>(i,j)[1],
                                                                                  destinationY.at<cv::Vec3b>(i,j)[2]
                                                                                  ),
                                                                   CalculMoyenne (destinationX.at<cv::Vec3b>(i,j)[0],
                                                                                  destinationX.at<cv::Vec3b>(i,j)[1],
                                                                                  destinationX.at<cv::Vec3b>(i,j)[2]
                                                                                  )
                                                               );
        }
    }
    
    destMulti0 = imageConvolution(image, multi0);
    destMulti1 = imageConvolution(image, multi1);
    destMulti2 = imageConvolution(image, multi2);
    destMulti3 = imageConvolution(image, multi3);
    
    cvtColor(destMulti0, grayMulti0, cv::COLOR_RGB2GRAY);
    cvtColor(destMulti1, grayMulti1, cv::COLOR_RGB2GRAY);
    cvtColor(destMulti2, grayMulti2, cv::COLOR_RGB2GRAY);
    cvtColor(destMulti3, grayMulti3, cv::COLOR_RGB2GRAY);

    distMulti = max(grayMulti0, grayMulti1);
    distMulti = max(distMulti, grayMulti2);
    distMulti = max(distMulti, grayMulti3);
    
    distMulti.convertTo(distMulti, CV_32F);
    
    dirMulti = directionMulti(distMulti, dirColorMulti, grayMulti0, grayMulti1, grayMulti2, grayMulti3);
    
    cv::Scalar mean, stddev;
    cv::meanStdDev(distMulti, mean, stddev, cv::Mat());
    printf("%g  %g\n", mean[0], stddev.val[0]/255);
    
    seuil = mean.val[0]/255;
    printf("%d\n",(int) (seuil * 100));
    seuil_slider = (int) (seuil * 100);
    seuil_hyst = 0.02f;
    
//    seuillage(destinationNorme, destinationNormeSeuil, seuil);
    seuillage(distMulti, distMultiSeuil, seuil);
    
    GrisChan[0] = destinationNorme/255;
    GrisChan[1] = destinationNorme/255;
    GrisChan[2] = destinationNorme/255;
    merge(GrisChan, 3, destinationNormeGris);
    
    /// Create Window
    cv::namedWindow("gradientNormeSeuil", 1);
    cv::namedWindow("Hysteresis", 1);
    
    /// Create Trackbars
    char TrackbarThresholdName[50];
    sprintf( TrackbarThresholdName, "seuil");
    char TrackbarHystName[50];
    sprintf( TrackbarHystName, "hysteresis");
    
    cv::createTrackbar( TrackbarThresholdName, "gradientNormeSeuil", &seuil_slider, seuil_slider_max, on_trackbar );
    cv::createTrackbar( TrackbarHystName, "Hysteresis", &seuil_hyst_slider, seuil_hyst_slider_max, on_trackbar );
    
    seuil_hysteresis (destinationNorme, seuilHaut, seuilHyst, seuil, seuil_hyst);
    /// Show some stuff
    on_trackbar( seuil_slider, 0 );
    on_trackbar( seuil_hyst_slider, 0 );
    

    
//    cv::imshow("image", image);
//
//    cv::imshow("gradX", destinationX);
//    cv::imshow("gradY", destinationY);

//    cv::imshow("multi0", destMulti0);
//    cv::imshow("multi1", destMulti1);
//    cv::imshow("multi2", destMulti2);
//    cv::imshow("multi3", destMulti3);
//    cv::imshow("gray0", grayMulti0);
    cv::imshow("dist", distMulti);
    cv::imshow("dir", dirMulti);
    cv::imshow("dirColor", dirColorMulti);
    
//
//    cv::imshow("gradientNormeGris", destinationNormeGris);
//    cv::imshow("gradientDirection", destinationDirection);

    
    cv::waitKey(0);
    
    return 0;
}



