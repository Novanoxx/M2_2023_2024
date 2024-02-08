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

int main(int argc, char** argv )
{
    std::string folder("../img_normalized/*.png");
    std::vector<String> filenames;
    cv::glob(folder, filenames, false);
    std::vector<Mat> imgVector;

    for(size_t i = 0; i < filenames.size(); ++i)
    {
        Mat img = imread(filenames[i], IMREAD_GRAYSCALE);
        if(!img.data)
        {
            std::cerr << "Problem loading image" << std::endl;
            return -1;
        }
        normalize(img, img, 0, 65535.0, NORM_MINMAX, CV_16U);

        // // to see the image with opencv
        // namedWindow("Display Image", WINDOW_AUTOSIZE );
        // imshow("Display Image", img);
        // waitKey(0);
        imgVector.push_back(img);
    }

    // Create Background Subtractor objects
    Ptr<BackgroundSubtractor> pBackSub = createBackgroundSubtractorMOG2();

    Mat fgMask;
    for (Mat imgSrc : imgVector)
    {
        //update the background model
        pBackSub->apply(imgSrc, fgMask);

        //show the current frame and the fg masks
        imshow("Frame", imgSrc);
        imshow("FG Mask", fgMask);

        //get the input from the keyboard
        int keyboard = waitKey(0);
        if (keyboard == 'q' || keyboard == 27)
            break;
    }
    
    return 0;
}