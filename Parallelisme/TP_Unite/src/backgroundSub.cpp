#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
namespace fs = std::filesystem;

// https://docs.opencv.org/3.4/d1/dc5/tutorial_background_subtraction.html
// https://stackoverflow.com/questions/36004948/segmentation-of-foreground-from-background

bool myCmp(std::string s1, std::string s2)
{
 
    if (s1.size() == s2.size())
    {
        return s1 < s2;
    }
    else
    {
        return s1.size() < s2.size();
    }
}

int main(int argc, char** argv )
{
    std::string folder("../img_normalized/*.png");
    std::vector<String> filenames;
    cv::glob(folder, filenames, false);
    std::vector<Mat> imgVector;

    std::sort(filenames.begin(), filenames.end(), myCmp);

    for(size_t i = 0; i < filenames.size(); ++i)
    {
        Mat img = imread(filenames[i], IMREAD_GRAYSCALE);
        if(!img.data)
        {
            std::cerr << "Problem loading image" << std::endl;
            return -1;
        }
        equalizeHist(img, img);

        imgVector.push_back(img);
    }

    // Create Background Subtractor objects
    Ptr<BackgroundSubtractor> pBackSubMOG2 = createBackgroundSubtractorMOG2();
    Ptr<BackgroundSubtractor> pBackSubKNN = createBackgroundSubtractorKNN();

    Mat fgMask, fgMask2, labels, stats, centroids;
    int type = MORPH_ELLIPSE;
    Mat erodeElement = getStructuringElement( type,
                       Size( 5, 5 ));
    Mat dilateElement = getStructuringElement( type,
                       Size( 20, 20 ));
    
    // for training the mask (remove the background on the first frame)
    for (Mat imgSrc : imgVector)
    {
        //update the background model
        pBackSubMOG2->apply(imgSrc, fgMask);
        pBackSubKNN->apply(imgSrc, fgMask2);
    }

    for (Mat imgSrc : imgVector)
    {
        //update the background model
        pBackSubMOG2->apply(imgSrc, fgMask);
        pBackSubKNN->apply(imgSrc, fgMask2);

        erode(fgMask, fgMask, erodeElement);
        dilate(fgMask, fgMask, dilateElement);

        // threshold(fgMask, fgMask, 200, 0, THRESH_TOZERO);

        erode(fgMask2, fgMask2, erodeElement);
        dilate(fgMask2, fgMask2, dilateElement);

        // threshold(fgMask2, fgMask2, 200, 0, THRESH_TOZERO);

        int nbLabel = connectedComponentsWithStats(fgMask2, labels, stats, centroids, 8, CV_16U);
        // std::cout << nbLabel << std::endl;
        //Mat tmp = Mat(fgMask2.size(), , ); //init with scalar 00 fgmask2.size;

        for (int i = 0; i < nbLabel; i++)
        {
            if (stats.at<int>(i, 4) > 400)
            {
                nbPerson += 1;
                //tmp = tmp | labels.at<int>(i);
            }
        }

        //show the current frame and the fg masks
        std::cout << nbPerson << std::endl;
        imshow("Frame", imgSrc);
        imshow("FG Mask MOG2", fgMask);
        imshow("FG Mask KNN", fgMask2);

        //get the input from the keyboard
        int keyboard = waitKey(100);
        if (keyboard == 'q' || keyboard == 27)
            break;
    }
    
    return 0;
}