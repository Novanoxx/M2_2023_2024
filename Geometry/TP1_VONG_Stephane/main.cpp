#include <iostream>
#include <DGtal/base/Common.h>
#include <DGtal/helpers/StdDefs.h>
#include <DGtal/images/ImageSelector.h>
#include <DGtal/images/imagesSetsUtils/SetFromImage.h>
#include "DGtal/io/readers/PGMReader.h"
#include "DGtal/io/writers/GenericWriter.h"
#include <DGtal/io/boards/Board2D.h>
#include <DGtal/io/colormaps/ColorBrightnessColorMap.h>
#include "DGtal/io/Color.h"
#include <DGtal/topology/SurfelAdjacency.h>
#include <DGtal/topology/helpers/Surfaces.h>
#include "DGtal/topology/KhalimskySpaceND.h"
#include "DGtal/geometry/curves/ArithmeticalDSSComputer.h"
#include "DGtal/geometry/curves/FreemanChain.h"
#include "DGtal/geometry/curves/GreedySegmentation.h"

#define _USE_MATH_DEFINES

using namespace std;
using namespace DGtal;
using namespace Z2i;

typedef FreemanChain<int> Contour4; 
typedef ArithmeticalDSSComputer<Contour4::ConstIterator,int,4> DSS4;
typedef GreedySegmentation<DSS4> Decomposition4;

template<class T>
Curve getBoundary(T & object, DigitalSet set)
{
    //Khalimsky space
    KSpace kSpace;
    // we need to add a margine to prevent situations such that an object touch the bourder of the domain
    /*
    kSpace.init( object.domain().lowerBound() - Point(1,1),
                   object.domain().upperBound() + Point(1,1), true );
    */
    kSpace.init( set.domain().lowerBound() - Point(1,1),
                   set.domain().upperBound() + Point(1,1), true );
    // 1) Call Surfaces::findABel() to find a cell which belongs to the border
    std::vector<Z2i::Point> boundaryPoints; // boundary points are going to be stored here
    auto set2d = object.pointSet();
    auto aCell = Surfaces<Z2i::KSpace>::findABel(kSpace, set2d, 100000);
    
    // 2) Call Surfece::track2DBoundaryPoints to extract the boundary of the object
    Curve boundaryCurve;
    SurfelAdjacency<2> SAdj( true );
    Surfaces<Z2i::KSpace>::track2DBoundaryPoints(boundaryPoints, kSpace, SAdj, set2d, aCell);
    // 3) Create a curve from a vector
    boundaryCurve.initFromPointsVector(boundaryPoints);
    return boundaryCurve;
}

template<class T>
void sendToBoard( Board2D & board, T & p_Object, DGtal::Color p_Color) {
    board << CustomStyle( p_Object.className(), new DGtal::CustomFillColor(p_Color));
    board << p_Object;
}

double circularity(double perimeter, double area)
{
    // roundness = (perimeter * perimeter) / (4 * pi * area)
    return pow(perimeter, 2) / (4 * M_PI * area);
}

double getAverage(vector<double> average)
{
    double ret = 0;
    for (int i = 0; i < average.size(); i++)
    {
        ret += average.at(i);
    }
    return ret/average.size();
}

double getStandardDeviation(vector<double> values)
{
    double ret = 0;
    double averageCal = getAverage(values);
    for (int i = 0; i < values.size(); i++)
    {
        ret += abs(values.at(i) - averageCal);
    }
    return ret/values.size();
}

template<class T>
std::array<double, 2> step4_5_6(Board2D & aBoard, Curve boundaryCurve, T digitalObject) {
    // Step 4
    stringstream freemanChain(stringstream::in | stringstream::out);
    string codeRange;
    for (auto it = boundaryCurve.getCodesRange().begin(); 
        it != boundaryCurve.getCodesRange().end();
        it++)
    {
        codeRange += *it;
    }
    auto firstPixel = *(boundaryCurve.getPointsRange().begin());

    freemanChain << firstPixel[0] << " " << firstPixel[1] << " " << codeRange << endl;
    // std::cout << freemanChain.str() << std::endl;
    Contour4 contour(freemanChain);

    Decomposition4 decompose(contour.begin(), contour.end(), DSS4());

    aBoard << SetMode("PointVector", "Both");
    aBoard << SetMode(boundaryCurve.className(), "Points") 
            << boundaryCurve;

    for ( Decomposition4::SegmentComputerIterator 
            it = decompose.begin(),
            itEnd = decompose.end();
            it != itEnd; ++it ) 
        {
        aBoard << SetMode( "ArithmeticalDSS", "Points" )
                << it->primitive();
        }

    for ( Decomposition4::SegmentComputerIterator 
            it = decompose.begin(),
            itEnd = decompose.end();
            it != itEnd; ++it ) 
        {
        aBoard << SetMode( "ArithmeticalDSS", "BoundingBox" )
                << CustomStyle( "ArithmeticalDSS/BoundingBox", 
                                new CustomPenColor( Color::Blue ) )
                << it->primitive();
        // auto a = *(it.begin());
        // std::cout << "full : " << a << " x : " << a[0] << " y : " << a[1] << std::endl;
        }

    // Step 5
    int area = 0;
    vector<Integer> xPointArray;
    vector<Integer> yPointArray;

    for ( Decomposition4::SegmentComputerIterator 
            it = decompose.begin(),
            itEnd = decompose.end();
            it != itEnd; ++it ) 
        {
            auto a = *(it.begin());
            xPointArray.push_back(it.begin().get()[0]);
            yPointArray.push_back(it.begin().get()[1]);
        }
    int arraySize = xPointArray.size();

    for (int i = 0; i < arraySize; i++)
    {
        if (i == arraySize - 1)
        {
            area += (xPointArray[i] * yPointArray[0]) - (xPointArray[0] * yPointArray[i]);
        } else {
            area += (xPointArray[i] * yPointArray[i + 1]) - (xPointArray[i + 1] * yPointArray[i]);
        }
    }

    cout << "-----------------STEP 5-----------------" << endl;
    cout << "Number of 2-Cells : " << digitalObject.size() << endl;
    cout << "Polygon area size : " << area/2 << endl;

    // Step 6
    double perimeter = 0;
    for (int i = 1; i < arraySize; i++)
    {
        if (i == arraySize - 1)
        {
            perimeter += sqrt( (pow(xPointArray[0] - xPointArray[i], 2) + pow(yPointArray[0] - yPointArray[i], 2)));
        } else {
            perimeter += sqrt( (pow(xPointArray[i] - xPointArray[i - 1], 2) + pow(yPointArray[i] - yPointArray[i - 1], 2)));
        }
    }
    cout << "-----------------STEP 6-----------------" << endl;
    cout << "Number of 1-Cells : " << boundaryCurve.size() << endl;
    cout << "Polygon perimeter size : " << perimeter << endl;
    
    cout << "-----------------STEP 7-----------------" << endl;
    cout << "Circularity : " << circularity(perimeter, area) << endl << endl;

    return std::array<double, 2> {{area/2, perimeter}};
}

int main(int argc, char** argv)
{
    setlocale(LC_NUMERIC, "us_US"); //To prevent French local settings
    typedef ImageSelector<Domain, unsigned char >::Type Image; // type of image
    typedef DigitalSetSelector< Domain, BIG_DS+HIGH_BEL_DS >::Type DigitalSet; // Digital set type
    //typedef Object<DT8_4, DigitalSet> ObjectType; // Digital object type
    typedef Object<DT4_8, DigitalSet> ObjectType; // Digital object type


    // read an image
    Image image = PGMReader<Image>::importPGM ("../RiceGrains/Rice_japonais_seg_bin.pgm"); // you have to provide a correct path as a parameter
    //Image image = PGMReader<Image>::importPGM ("../RiceGrains/Rice_camargue_seg_bin.pgm");
    //Image image = PGMReader<Image>::importPGM ("../RiceGrains/Rice_basmati_seg_bin.pgm");

    // 1) make a "digital set" of proper size
    DigitalSet set2d (image.domain());

    // 2) populate a digital set from the image using SetFromImage::append()
    SetFromImage<DigitalSet>::append<Image>(set2d, image, 1, 255);

    // 3) Create a digital object from the digital set
    ObjectType digitalObject(dt4_8, set2d);
    //ObjectType digitalObject(dt8_4, set2d);
    
    vector< ObjectType > objects;          // All connected components are going to be stored in it
    back_insert_iterator< std::vector< ObjectType > > inserter( objects ); // Iterator used to populated "objects".

    // 4) Set the adjacency pair and obtain the connected components using "writeComponents"
    uint nbc = digitalObject.writeComponents(inserter);
    cout << "number of components : " << objects.size() << endl << endl; // Right now size of "objects" is the number of conected components

    // This is an example how to create a pdf file for each object
    Board2D aBoard;                                 // use "Board2D" to save output
    vector<double> averageArea;
    vector<double> averagePerimeter;
    vector<double> averageCircularity;
    for(auto it = objects.begin(); it != objects.end(); ++it)
    {
        digitalObject = *(it);
        auto boundaryCurve = getBoundary(digitalObject, set2d);
        sendToBoard(aBoard, boundaryCurve, Color::Red);
        auto result = step4_5_6(aBoard, boundaryCurve, digitalObject);
        averageArea.push_back(result[0]);
        averagePerimeter.push_back(result[1]);
        averageCircularity.push_back(circularity(result[1], result[0]));
    }
    
    cout << "Average area and standard deviation : " << getAverage(averageArea) << " +/- " << getStandardDeviation(averageArea) << endl;
    cout << "Average perimeter and standard deviation : " << getAverage(averagePerimeter) << " +/- " << getStandardDeviation(averagePerimeter) << endl;
    cout << "Average circularity and standard deviation : " << getAverage(averageCircularity) << " +/- " << getStandardDeviation(averageCircularity) << endl << endl;

    //aBoard.saveEPS("out.eps");
    #ifdef WITH_CAIRO
        aBoard.saveCairo("out.pdf",Board2D::CairoPDF); // do not forget to change the path!
    #endif

    return 0;
}
