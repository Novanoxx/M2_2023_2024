#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
namespace fs = std::filesystem;

int main(int argc, char** argv )
{
    if ( argc != 2 )
    {
        printf("usage: main <number>\n");
        return -1;
    }

    std::string path;
    char *string_part;

    switch(strtol(argv[1], &string_part, 10))
    {
        case 0:
            path = "../img/CROSS_X-F1-B1_P880043_20200625111459_225";
            break;
        case 1:
            path = "../img/FLOOR_Y-F1-B0_P220533_20201203141353_275";
            break;
        case 2:
            path = "../img/PASS-2LBO_X-F3-B0_P848995_20210303145112_275";
            break;
        default:
            printf("Can't find folder. \n");  
            break;   
    }

    if (!path.empty())
    {
        std::string folder(path + "/*.png");
        std::vector<String> filenames;
        cv::glob(folder, filenames, false);

        fs::create_directories("../img_normalized");
        for(size_t i = 0; i < filenames.size(); ++i)
        {
            Mat img = imread(filenames[i], IMREAD_GRAYSCALE);
            if(!img.data)
            {
                std::cerr << "Problem loading image" << std::endl;
                return -1;
            }
            normalize(img, img, 0, 65535.0, NORM_MINMAX, CV_16U);
            imwrite("../img_normalized/Normalized_" + std::to_string(i) + ".png", img);

            // // to see the image with opencv
            // namedWindow("Display Image", WINDOW_AUTOSIZE );
            // imshow("Display Image", img);
            // waitKey(0);
        }
    }
    
    return 0;
}