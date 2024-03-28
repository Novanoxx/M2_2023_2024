#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <sys/time.h>
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

void processImage(const Mat &imgSrc, Ptr<BackgroundSubtractor> pBackSubKNN, std::vector<int> &results, std::vector<Mat> &processedImages, std::vector<Mat> &processedMask) {
    Mat fgMask, labels, stats, centroids;
    int type = MORPH_ELLIPSE;
    Mat erodeElement = getStructuringElement(type, Size(5, 5));
    Mat dilateElement = getStructuringElement(type, Size(20, 20));

    // Mettre à jour le modèle de fond
    pBackSubKNN->apply(imgSrc, fgMask);

    erode(fgMask, fgMask, erodeElement);
    dilate(fgMask, fgMask, dilateElement);

    int nbLabel = connectedComponentsWithStats(fgMask, labels, stats, centroids, 8, CV_16U);
    int nbPerson = 0;

    for (int i = 0; i < nbLabel; i++) {
        if (stats.at<int>(i, 4) > 800) {
            nbPerson += 1;
        }
    }
    // Enregistrer le résultat
    results.push_back(nbPerson - 1);

    // Enregistrer l'image traitée
    processedImages.push_back(imgSrc.clone());
    processedMask.push_back(fgMask.clone());
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

    timeval begin, end;
    // Create Background Subtractor objects
    Ptr<BackgroundSubtractor> pBackSubKNN = createBackgroundSubtractorKNN();

    std::vector<int> results(imgVector.size());
    std::vector<Mat> processedImages;
    std::vector<Mat> processedMask;
    std::vector<suseconds_t> microTime;
    
    // Training
    #pragma omp parallel for
    for (size_t i = 0; i < imgVector.size(); ++i) {
        gettimeofday(&begin, nullptr);

        processImage(imgVector[i], pBackSubKNN, results, processedImages, processedMask);

        gettimeofday(&end, nullptr);
        suseconds_t time = end.tv_usec - begin.tv_usec;
        //std::cout << "MicroSeconds : " << time << std::endl;
        microTime.push_back(time);
    }

    results.clear();
    processedImages.clear();
    processedMask.clear();
    microTime.clear();

    #pragma omp parallel for
    for (size_t i = 0; i < imgVector.size(); ++i) {
        gettimeofday(&begin, nullptr);

        processImage(imgVector[i], pBackSubKNN, results, processedImages, processedMask);

        gettimeofday(&end, nullptr);
        suseconds_t time = end.tv_usec - begin.tv_usec;
        //std::cout << "MicroSeconds : " << time << std::endl;
        microTime.push_back(time);
    }
    suseconds_t res = 0;
    for (auto time : microTime)
    {
        res += time;
    }
    std::cout << "MicroSeconds Total : " << res << std::endl;

    auto font = FONT_HERSHEY_SIMPLEX;
    //show the current frame and the fg masks
    for (size_t i = 0; i < processedImages.size(); ++i) {
        putText(processedMask[i], "Nombre de composantes connexes: " + std::to_string(results[i]), Point(20, 50), font, 0.75, Scalar(255., 255., 255.), 1.5, LINE_AA);
        imshow("ImageSrc", processedImages[i]);
        imshow("ImageMaskKNN", processedMask[i]);

        //get the input from the keyboard
        int keyboard = waitKey(0);
        if (keyboard == 'q' || keyboard == 27)
            break;
    }
    
    return 0;
}