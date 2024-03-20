#include <iostream>
#include <cstdlib>
#include <list>
#include <c3ga/Mvec.hpp>

#include "c3gaTools.hpp"
#include "Geogebra_c3ga.hpp"

#include "Entry.hpp"




///////////////////////////////////////////////////////////////
void sphere_sphere_intersection(){

    Viewer_c3ga viewer;
    viewer.showAxis(false);
    
    // a primal sphere
    c3ga::Mvec<double> sphere1 = c3ga::point<double>(-2,0,0.5)
                              ^ c3ga::point<double>(-1,1,0.5)
                              ^ c3ga::point<double>(-1,-1,0.5)
                              ^ c3ga::point<double>(-1,0,1.5);
    viewer.push(sphere1, "sphere1", 0,0,200);
    std::cout << "S1 " << sphere1 << std::endl;

    // a primal sphere
    c3ga::Mvec<double> sphere2 = c3ga::point<double>(-1.5,0,0.5)
                              ^ c3ga::point<double>(-0.5,1,0.5)
                              ^ c3ga::point<double>(-0.5,-1,0.5)
                              ^ c3ga::point<double>(-0.5,0,1.5);
    viewer.push(sphere2, "sphere2", 200,0,0);
    std::cout << "S2 " << sphere2 << std::endl;

    // circle
    c3ga::Mvec<double> circle = (sphere1.dual() ^ sphere2.dual()).dual();
    viewer.push(circle, "circle", 0,200,0);
    std::cout << "C " << circle << std::endl;

    viewer.display();
    viewer.render("output.html");
}


///////////////////////////////////////////////////////////////
void line_throw_circle(){

    Viewer_c3ga viewer;
    viewer.showAxis(false);
    viewer.showGrid(false);

    // a primal circle
    c3ga::Mvec<double> C = c3ga::point<double>(2, 0,0.5)
                         ^ c3ga::point<double>(2, 2,2.5)
                         ^ c3ga::point<double>(2,-2,2.5);

    viewer.push(C, "C", 0,0,200);
    std::cout << "C " << C << std::endl;

    // generate some lines
    c3ga::setRandomSeed();
    const unsigned int nbLine = 10;
    for(unsigned int i=0; i<nbLine; ++i){
    	c3ga::Mvec<double> L = c3ga::randomPoint<double>() 
                             ^ c3ga::randomPoint<double>() 
                             ^ c3ga::ei<double>();

        c3ga::Mvec<double> dual_S = L.dual() | C;
        double radius = dual_S | dual_S;

        if(radius < 0.0) {
            viewer.push(L, 200,0,0);
            //viewer.push(dual_S,"S",100,100,200);  // display the dual sphere
        }
        else viewer.push(L, 0,200,0);
    }

    viewer.display();
    viewer.render("output.html");
}


///////////////////////////////////////////////////////////////
void translation(){

    Viewer_c3ga viewer;

    // a primal circle
    c3ga::Mvec<double> C = c3ga::point<double>(2, 0,0.5)
                         ^ c3ga::point<double>(2, 2,2.5)
                         ^ c3ga::point<double>(2,-2,2.5);
    viewer.push(C, "C", 0,0,200);
    std::cout << "C " << C << std::endl;

    // define a translator
    const c3ga::Mvec<double> t = 0.8*c3ga::e1<double>(); // translation vector
    c3ga::Mvec<double> T = 1 - 0.5*t*c3ga::ei<double>(); // translator

    //  translate the circle many times
    const unsigned int nbCircles = 10;
    for(unsigned int i=0; i<nbCircles; ++i){

    	C = T * C / T;
		viewer.push(C, "C", 0,0,200);
    	std::cout << "C " << C << std::endl;
    }

    viewer.display();
    viewer.render("output.html");
}


///////////////////////////////////////////////////////////////
void rotation(){

    Viewer_c3ga viewer;

    // a primal circle
    c3ga::Mvec<double> C = c3ga::point<double>(1, 0,1.5)
                         ^ c3ga::point<double>(3, 0,1.5)
                         ^ c3ga::point<double>(2, 0,0.5);
    viewer.push(C, "C", 200,0,0);
    std::cout << "C " << C << std::endl;

    // nb circles
	const unsigned int nbCircles = 30;
	const double angle = 2*M_PI / nbCircles; // in radian

    // define a rotor
    c3ga::Mvec<double> rotationPlane = c3ga::e12<double>();
    c3ga::Mvec<double> R = cos(0.5*angle) - rotationPlane * sin(0.5*angle);

    // creates rotated circles
    for(unsigned int i=0; i<nbCircles-1; ++i){

    	C = R * C / R;
		viewer.push(C, "C", 0,0,200);
    	std::cout << "C " << C << std::endl;
    }

    viewer.display();
    viewer.render("output.html");
}


///////////////////////////////////////////////////////////////
void dilator(){

    Viewer_c3ga viewer;
    viewer.showAxis(true);
    viewer.showGrid(false);

    // a primal circle
    c3ga::Mvec<double> C = c3ga::point<double>(5, 2,2.5)
                         ^ c3ga::point<double>(5, 0,0.5)
                         ^ c3ga::point<double>(5,-2,2.5);
    viewer.push(C, "C", 0,0,200);
    std::cout << "C " << C << std::endl;

    // define a dilator
	const double scale = 0.8; 
    c3ga::Mvec<double> S = 1 - c3ga::e0i<double>()*(1.0-scale)/(1.0+scale);

    // nb circles
	const unsigned int nbCircles = 10;
    for(unsigned int i=0; i<nbCircles; ++i){

    	C = S * C / S;
		viewer.push(C, "C", 0,0,200);
    	std::cout << "C " << C << std::endl;
    }

    viewer.display();
    viewer.render("output.html");
}


///////////////////////////////////////////////////////////////
void coquillage(){

	// coquillage parameters
	const unsigned int nbIter = 100;
	const unsigned int nbTours = 5;
	const double size = 10.0;

	// rotor
	const double angle = 2*M_PI * nbTours/ (double)nbIter; // in radian
    c3ga::Mvec<double> axis = c3ga::e12<double>();
    c3ga::Mvec<double> R = cos(0.5*angle) - axis * sin(0.5*angle);

    // translator
    c3ga::Mvec<double> T = 1 - 0.5*(size/(double)nbIter) * c3ga::e3<double>()*c3ga::ei<double>();

    // dilator
    const double scale = 1.0/std::pow(10, 1.0/nbIter);
	c3ga::Mvec<double> D = 1 - c3ga::e0i<double>()*(1.0-scale)/(1.0+scale);

	// motor
	c3ga::Mvec<double> M = T * R * D;
	std::cout << "motor " << M << std::endl;

	// sphere
    c3ga::Mvec<double> sphere = c3ga::point<double>(-2,0,0.5)
                              ^ c3ga::point<double>(-1,1,0.5)
                              ^ c3ga::point<double>(-1,-1,0.5)
                              ^ c3ga::point<double>(-1,0,1.5);
                              
    // c3ga::Mvec<double> sphere = c3ga::point<double>(1, 0,1.5)
    //                      ^ c3ga::point<double>(3, 0,1.5)
    //                      ^ c3ga::point<double>(2, 0,0.5);                              


    Viewer_c3ga viewer;
	viewer.push(sphere, 150,0,0);


	// render
    for(unsigned int i=0; i<nbIter; ++i){

    	sphere = M * sphere / M;
    	std::cout << sphere << std::endl;
    	sphere.roundZero(1.0e-10); //sphere = sphere.grade(4);
		viewer.push(sphere, 150,0,0);
    }

    viewer.display();
    viewer.render("output.html");
}


///////////////////////////////////////////////////////////////
void point_pairs(){
    Viewer_c3ga viewer;
    viewer.showAxis(false);

    // 3  points on plane z=0.1
    c3ga::Mvec<double> pt1 = c3ga::point<double>(1,2,0.1);
    c3ga::Mvec<double> pt2 = c3ga::point<double>(3,2,0.1);
    c3ga::Mvec<double> pt3 = c3ga::point<double>(2,3,0.1);

    // plane z=0
    c3ga::Mvec<double> plane = pt1 ^ pt2 ^ pt3 ^ c3ga::ei<double>();
    viewer.push(plane, 50,50,50);

    // sphere from 4 points
    c3ga::Mvec<double> sphere = c3ga::point<double>(0,0,2.0) 
                              ^ c3ga::point<double>(2,0,2.0) 
                              ^ c3ga::point<double>(1,1,2.0) 
                              ^ c3ga::point<double>(1,0,3.0);
    viewer.push(sphere, 200,200,00);

    // circle from 3 points
    c3ga::Mvec<double> circle = c3ga::point<double>(0,-0.75,0.5) 
                              ^ c3ga::point<double>(2,-0.75,0.5) 
                              ^ c3ga::point<double>(1,-0.75,1.5);
    viewer.push(circle, 200, 0, 0);

    // line from 2 points
    c3ga::Mvec<double> line = c3ga::point<double>(0,-0.75,2.3) 
                              ^ c3ga::point<double>(2,-0.75,2.3) 
                              ^ c3ga::ei<double>();
    viewer.push(line, 200, 0, 0);

    // point pairs from cicle and plane
    c3ga::Mvec<double> pp1 = !circle ^ !plane;
    viewer.push(!pp1, 0, 200, 0);

    // point pairs from cicle and sphere
    c3ga::Mvec<double> pp2 = !circle ^ !sphere;
    viewer.push(!pp2, 0, 200, 0);

    // point pairs from line and sphere
    c3ga::Mvec<double> pp3 = !line ^ !sphere;
    viewer.push(!pp3, 0, 200, 0);


    viewer.display();
    viewer.render("output.html");
}


///////////////////////////////////////////////////////////////
void primal_objects()
{
	// 4 points
    c3ga::Mvec<double> pt1 = c3ga::point<double>(1,2,0.5);
    c3ga::Mvec<double> pt2 = c3ga::point<double>(3,2,0.5);
    c3ga::Mvec<double> pt3 = c3ga::point<double>(2,3,0.5);
    c3ga::Mvec<double> pt4 = c3ga::point<double>(2,2,2.0);

    Viewer_c3ga viewer;
    viewer.showAxis(false);
    viewer.push(pt1, "pt1", 200,0,0);
    viewer.push(pt2, "pt2", 200,0,0);
    viewer.push(pt3, "pt3", 200,0,0);
    viewer.push(pt4, "pt4", 200,0,0);

    // a primal line
    c3ga::Mvec<double> line = pt3 ^ pt2 ^ c3ga::ei<double>();
    viewer.push(line, "line", 0,200,0);

    // a primal circle
    c3ga::Mvec<double> circle = pt1 ^ pt2 ^ pt3;
    viewer.push(circle, "circle", 0,200,0);

    // a primal plane
    c3ga::Mvec<double> plane = pt1 ^ pt2 ^ pt3 ^ c3ga::ei<double>();
    viewer.push(plane, "plane", 150,200,0);

    // a primal sphere
    c3ga::Mvec<double> sphere = pt1 ^ pt2 ^ pt3 ^ pt4;
    viewer.push(sphere, "sphere", 0,0,200);
    std::cout << "sphere " << sphere << std::endl;

    viewer.display();
    viewer.render("output.html");
}


///////////////////////////////////////////////////////////////
void intersections()
{
    Viewer_c3ga viewer;
    viewer.showAxis(false);

    // a sphere from 4 points
    c3ga::Mvec<double> pt1 = c3ga::point<double>(1,2,0.5);
    c3ga::Mvec<double> pt2 = c3ga::point<double>(3,2,0.5);
    c3ga::Mvec<double> pt3 = c3ga::point<double>(2,3,0.5);
    c3ga::Mvec<double> pt4 = c3ga::point<double>(2,2,2.0);
    c3ga::Mvec<double> sphere = pt1 ^ pt2 ^ pt3 ^ pt4;
    viewer.push(sphere, "sphere", 0,0,200);

    // a line (for sphere intersection)
    c3ga::Mvec<double> pt5 = c3ga::point<double>(3,2,1.5);
    c3ga::Mvec<double> pt6 = c3ga::point<double>(2,3,1.5);
    c3ga::Mvec<double> line = pt5 ^ pt6 ^ c3ga::ei<double>();
    viewer.push(line, "line", 200,0,0);

    // pair point (line-sphere intersection)
    c3ga::Mvec<double> pp = (line.dual() ^ sphere.dual()).dual();
    viewer.push(pp, "", 0,200,0);
  
    // a plane from 3 points
    c3ga::Mvec<double> plane = pt1 ^ pt2 ^ pt3 ^ c3ga::ei<double>();
    viewer.push(plane, "plane", 150,200,0);

    // a line from 2 points (for plane intersection)
    c3ga::Mvec<double> pt7 = c3ga::point<double>(-3,2,1.5);
    c3ga::Mvec<double> pt8 = c3ga::point<double>(-5,4,0);
    c3ga::Mvec<double> l2 = pt7 ^ pt8 ^ c3ga::ei<double>();
    viewer.push(l2, "l2", 200,0,0);

    // flat point (line-plane intersection)
    c3ga::Mvec<double> fp = (l2.dual() ^ plane.dual()).dual();
    viewer.push(fp, "", 0,200,0);

    // circle from 3 points (for plane intersection)
    c3ga::Mvec<double> pt9  = c3ga::point<double>(-3,-2,0.5);
    c3ga::Mvec<double> pt10 = c3ga::point<double>(-2,-3,0.5);
    c3ga::Mvec<double> pt11 = c3ga::point<double>(-2.5,-2.5,2.0);
    c3ga::Mvec<double> circle1 = pt9 ^ pt10 ^ pt11; 
    viewer.push(circle1, 200,0,0);

    // pair point (circle plane intersection)
    c3ga::Mvec<double> pp2 = (circle1.dual() ^ plane.dual()).dual();
    viewer.push(pp2, 0,200,0);

    // circle fro m3 points (for sphere intersection)
    c3ga::Mvec<double> pt12 = c3ga::point<double>(2,2,1);
    c3ga::Mvec<double> pt13 = c3ga::point<double>(0,1,1);
    c3ga::Mvec<double> pt14 = c3ga::point<double>(1,0,1);
    c3ga::Mvec<double> circle2 = pt12 ^ pt13 ^ pt14; 
    viewer.push(circle2, 200,0,0);

    // pair point (circle sphere intersection)
    c3ga::Mvec<double> pp3 = (circle2.dual() ^ sphere.dual()).dual();
    viewer.push(pp3, 0,200,0);

    // plane from 3 points(for plane intersection)
    c3ga::Mvec<double> pt15 = c3ga::point<double>(-5,-3,0.5);
    c3ga::Mvec<double> pt16 = c3ga::point<double>(-3,-5,0.5);
    c3ga::Mvec<double> pt17 = c3ga::point<double>(-5,-7,5.0);
    c3ga::Mvec<double> plane2 = pt15 ^ pt16 ^ pt17 ^ c3ga::ei<double>();
    viewer.push(plane2, "plane2", 250,100,0);

    // line (plane-plane intersection)
    c3ga::Mvec<double> line2 = (plane.dual() ^ plane2.dual() );
    viewer.push(line2, "l2", 0,200,0);

    viewer.display();
    viewer.render("output.html");
}


///////////////////////////////////////////////////////////////
void polygons()
{
    Viewer_c3ga viewer;
    viewer.showAxis(false);

    // a set of points
    c3ga::Mvec<double> pt1 = c3ga::point<double>(1  , 2, 0.5);
    c3ga::Mvec<double> pt2 = c3ga::point<double>(2  , 3, 0.5);
    c3ga::Mvec<double> pt3 = c3ga::point<double>(3  , 3, 0.5);
    c3ga::Mvec<double> pt4 = c3ga::point<double>(4  , 2, 0.5);
    c3ga::Mvec<double> pt5 = c3ga::point<double>(2.5,-2, 0.5);

    // put points on a list
    std::list<c3ga::Mvec<double>> myList;
    myList.push_back(pt1);
    myList.push_back(pt2);
    myList.push_back(pt3);
    myList.push_back(pt4);
    myList.push_back(pt5);

    viewer.pushPolygon(myList, 0,200,0);

    viewer.display();
    viewer.render("output.html"); 
}


///////////////////////////////////////////////////////////////
void tangent()
{
    Viewer_c3ga viewer;
    viewer.showAxis(false);

    // a sphere
    c3ga::Mvec<double> pt1 = c3ga::point<double>(1,2,0.5);
    c3ga::Mvec<double> pt2 = c3ga::point<double>(3,2,0.5);
    c3ga::Mvec<double> pt3 = c3ga::point<double>(2,3,0.5);
    c3ga::Mvec<double> pt4 = c3ga::point<double>(2,2,2.0);
    c3ga::Mvec<double> sphere = pt1 ^ pt2 ^ pt3 ^ pt4;
    viewer.push(sphere, "sphere", 0,0,200);

    // a point on the sphere
    viewer.push(pt4, "pt", 200,200,0);

    // tangent vector
    c3ga::Mvec<double> tangent = (sphere | pt4).dual();
    viewer.push(tangent, "t", 200,0,0);

    // a plane
    c3ga::Mvec<double> plane = pt3 ^ pt2 ^ pt1 ^ c3ga::ei<double>();
    viewer.push(plane, "plane", 150,200,0);

    // a point of the plane
    c3ga::Mvec<double> pt7 = c3ga::point<double>(2,-3,0.5);
    viewer.push(pt7, "pt7", 0,200,0);

    // tangent vector
    c3ga::Mvec<double> tangent2 = (plane | pt7).dual();
    viewer.push(tangent2, "t2", 200,0,0);

    // a line 
    c3ga::Mvec<double> pt5 = c3ga::point<double>(-3,2,1.5);
    c3ga::Mvec<double> pt6 = c3ga::point<double>(-5,4,0);
    c3ga::Mvec<double> line = pt5 ^ pt6 ^ c3ga::ei<double>();
    viewer.push(line, 0,0,200);

    // a point of the plane
    c3ga::Mvec<double> pt8 = c3ga::point<double>(1,-2,4.5);
    viewer.push(pt8, 0,200,0);

    // tangent vector
    c3ga::Mvec<double> tangent3 = line | pt8;
    viewer.push(tangent3, "", 200,0,0);

    // circle
    c3ga::Mvec<double> pt9  = c3ga::point<double>(-2,-1,2);
    c3ga::Mvec<double> pt10 = c3ga::point<double>(-3,-2,2.5);
    c3ga::Mvec<double> pt11 = c3ga::point<double>(-2.5,-1,2);
    c3ga::Mvec<double> circle = pt9 ^ pt10 ^ pt11;
    viewer.push(circle, "", 0,0,200);

    // tangent circle
    c3ga::Mvec<double> tangentCircle = pt10 | circle;
    viewer.push(tangentCircle, 200,0,0);
    viewer.push(pt10, 0,200,0);


    viewer.display();
    viewer.render("output.html");
}



///////////////////////////////////////////////////////////////
int main(){


	//sphere_sphere_intersection();
	 line_throw_circle();

    // translation();
	// rotation();
    // dilator();
	// coquillage();

    // bonus
	// point_pairs();
	// primal_objects();
	// intersections();
	// polygons();
	// tangent();

    return 0;
}