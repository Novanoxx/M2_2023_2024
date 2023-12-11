# Counting and measuring grains

## Step 1

The caracteristic for each type of rice are :

- Basmati rice : A type of rice that is long and thin.
- Camargue rice : A type of rice that is medium size and medium width.
- Japanese rice : A type of rice that is round, is small size but large width.

## Step 2
````
+---------------------------+--------------+---------------+---------------+
| 							| Basmati Rice | Camargue Rice | Japanese Rice |
+---------------------------+--------------+---------------+---------------+
| Number of component (4_8) | 		   141 |           132 |           147 |
| Number of component (8_4) | 		   116 |           111 |           135 |
+---------------------------+--------------+---------------+---------------+
````
None of these images are well-composed because the number of component are not identical between the digital topology (4_8) and (8_4). An image is well-composed if there is the same number of component with the digital topology (4_8) and (8_4).

## Step 3
First, I create a topological set and add a new argument ***(DigitalSet set)*** for the fonction in order to have all the connected component
````cpp
kSpace.init( set.domain().lowerBound() - Point(1,1),
             set.domain().upperBound() + Point(1,1), true );
````

Then, I call the fonction Surfaces::findABel, with the third argument "10000" being the number of tries so as to not have an error, to pick a random grain
````cpp
std::vector<Z2i::Point> boundaryPoints; // boundary points are going to be stored here
auto set2d = object.pointSet();
auto aCell = Surfaces<Z2i::KSpace>::findABel(kSpace, set2d, 10000);
````

Last, I add the result gotten with **Surfaces\<Z2i::KSpace\>::track2DBoundaryPoints(boundaryPoints, kSpace, SAdj, set2d, aCell);** in the board, create a curve from the vector and return boundaryCurve that will be useful later
````cpp
Curve boundaryCurve;
SurfelAdjacency<2> SAdj( true );
Surfaces<Z2i::KSpace>::track2DBoundaryPoints(boundaryPoints, kSpace, SAdj, set2d, aCell);
boundaryCurve.initFromPointsVector(boundaryPoints);
return boundaryCurve;
````

I had in the main a for loop so as to get every grain of rice from the image
````cpp
for(auto it = objects.begin(); it != objects.end(); ++it)
{
    digitalObject = *(it);
    auto boundaryCurve = getBoundary(digitalObject, set2d);
    sendToBoard(aBoard, boundaryCurve, Color::Red);
}
````
Check the images in the folder Step3 in case you want to see the result of the function.

## Step 4
The function created for this step has 3 arguments : Board2D &board to get the board, Curve boundaryCurve to get the boundaryCurve gotten from the step 3 and T digitalObject to get the size of it (step 5).

First, I need to create a FreemanChain, I get the x and y of the first pixel with
````cpp
auto firstPixel = \*(boundaryCurve.getPointsRange().begin());
````
and the code range with
````cpp
stringstream freemanChain(stringstream::in | stringstream::out);
string codeRange;
for (auto it = boundaryCurve.getCodesRange().begin(); 
    it != boundaryCurve.getCodesRange().end();
    it++)
{
    codeRange += \*it;
}
````
to create a FreemanChain
````cpp
freemanChain << firstPixel[0] << " " << firstPixel[1] << " " << codeRange << endl;
````

Then, I create the border with it named **decompose**
````cpp
Contour4 contour(freemanChain);
Decomposition4 decompose(contour.begin(), contour.end(), DSS4());
````

Last, I draw a dot in the center of the 2-Cell while traveling through the point of **decompose** and send everything in the board
````cpp
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

````
and I draw the border
````cpp
for ( Decomposition4::SegmentComputerIterator 
        it = decompose.begin(),
        itEnd = decompose.end();
        it != itEnd; ++it ) 
    {
    aBoard << SetMode( "ArithmeticalDSS", "BoundingBox" )
            << CustomStyle( "ArithmeticalDSS/BoundingBox", 
                            new CustomPenColor( Color::Blue ) )
            << it->primitive();
    }
````

I add
``step4_5_6(aBoard, boundaryCurve, digitalObject);``
in the loop for in the main to use the function.

Check the images in the folder Step4 in case you want to see the result of the function.  
I am printing only one grain because it is easier to see the border and all the detail.

## Step 5
I add this part in the function made previously.
To use the Shoelace formula, I must first extract all the point to use their value
````cpp
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
````

I add every value in the variable **area** when calculating the value with the Shoelace formula
````cpp
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
area /= 2;
cout << "Number of 2-Cells : " << digitalObject.size() << endl;
cout << "Polygon area size : " << area << endl;
````

>Mention in the report whether those area measurements maintain the multigrid convergence or not, citing the theoretical and experimental results seen during the courses.

I don't know

>Once you calculate the areas for all grains in an image, observe the distribution of estimated
areas for each rice type, and analyze it. Can the area be a principal component for classifying
grains into different types? Answer those questions in the report.

Here are examples of what my program is printing :
````
+------------------------------------------+---------------------+--------------------+---------------------+
|                  Values                  |      Japanese       |      Camargue      |       Basmati       |
+------------------------------------------+---------------------+--------------------+---------------------+
| For one grain :                          |                     |                    |                     |
| -----Number of 2-Cells                   | 1564                | 1229               | 1503                |
| -----Polygon area size                   | 1547                | 1213               | 1453                |
| For all grain :                          |                     |                    |                     |
| -----Average area and standard deviation | 1990.95 +/- 170.812 | 2448.4 +/- 586.425 | 2086.47 +/- 427.241 |
+------------------------------------------+---------------------+--------------------+---------------------+
````
We can not use the area for classifying grains into different types because the value of Camargue can overlap the Basmati one. Same for the Basmati and the Japanese one because of the standard deviation.

## Step 6

````cpp
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
cout << "Number of 1-Cells : " << boundaryCurve.size() << endl;
cout << "Polygon perimeter size : " << perimeter << endl;
````

>Mention in the report whether those perimeter measurements maintain the multigrid convergence or not, citing the theoretical and experimental results seen during the previous
courses.

I don't know

>Once you calculate the perimeters for all grains in an image, observe the distribution of
estimated perimeters for each rice type, and analyze it. Can the perimeter be a principal
component for classifying grains into different types? Answer those questions in the report.

Here are examples of what my program is printing :

````
+-----------------------------------------------+---------------------+---------------------+---------------------+
|                    Values                     |      Japanese       |      Camargue       |       Basmati       |
+-----------------------------------------------+---------------------+---------------------+---------------------+
| For one grain :                               |                     |                     |                     |
| -----Number of 1-Cells                        | 184                 | 172                 | 172                 |
| -----Polygon perimeter size                   | 140.731             | 125.306             | 125.306             |
| For all grain                                 |                     |                     |                     |
| -----Average perimeter and standard deviation | 205.039 +/- 28.6568 | 205.039 +/- 28.6568 | 191.193 +/- 29.8764 |
+-----------------------------------------------+---------------------+---------------------+---------------------+
````
We can not use the perimeter for classifying grains into different types because of the same reasons than the area one.

## Step 7

>Propose a circularity definition (write a mathematical formulation in the report)

circularity = (perimeter * perimeter) / (4 \* pi \* area)

>For grains of each rice type, calculate their circularities following your definition

I code this function that I use in the function step4_5_6 after printing the perimeter
````cpp
double circularity(double perimeter, double area)
{
    // roundness = (perimeter * perimeter) / (4 * pi * area)
    return pow(perimeter, 2) / (4 * M_PI * area);
}

cout << "Circularity : " << circularity(perimeter, area) << endl << endl;
````
(M_PI is a constant from cmath.h that is equal to pi)

>Compare the results between different grains, and make a discussion on characterization of the grain shapes using your circularity measure

Here are examples of what my program is printing :
````
+-------------------------------------------------+-----------------------+----------------------+--------------------+
|                     Values                      |       Japanese        |       Camargue       |      Basmati       |
+-------------------------------------------------+-----------------------+----------------------+--------------------+
| For one grain :                                 |                       |                      |                    |
| -----Circularity                                | 0.509387              | 0.515047             | 0.680266           |
| For all grain                                   |                       |                      |                    |
| -----Average circularity and standard deviation | 1.05108 +/- 0.0554032 | 1.27152 +/- 0.153263 | 1.6926 +/- 0.25577 |
+-------------------------------------------------+-----------------------+----------------------+--------------------+
````
There is a decent gap between all the values that the circularity give us even with the standard deviation. These values are values that tend to go to 1, meaning that it is perfectly round, and the more the value is far to 1, lesser it is round. In the values for one grain, except for the Basmati one, there is not enough gap to tell if it is Japanese or Camargue (certainly the ones that are cut from the images). This is why we must check the circularity average of an image because it tell more about the type of grain.