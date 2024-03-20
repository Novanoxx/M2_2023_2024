#include "ball.hpp"
#include <stdio.h>
#include <iostream>

using namespace std;

Ball::Ball()
{
    red = green = blue = 0.0;
}

Ball::Ball(const Ball &a)
{
    centre = a.centre;
    velocity = a.velocity;
    red = a.red;
    green = a.green;
    blue = a.blue;
    radius = 0.3;
    mass = 2;
}

Ball& Ball::operator=(const Ball& a) // assigns all the x,y,z coordinates.
{
    centre = a.centre;
    velocity = a.velocity;
    red = a.red;
    green = a.green;
    blue = a.blue;
    radius = 0.3;
    mass = 2;
    return *this;
}

void Ball::initBall(urd& xposGen, urd& yposGen,urd& velGen, urd& colorGen, mt19937& ran)
{   //generate random velocity
    centre = c3ga::point<float>(xposGen(ran), yposGen(ran), 0);
    velocity = c3ga::point<float>(velGen(ran), velGen(ran), 0);
    red = colorGen(ran);
    radius = 0.3;
    mass = 2;
    green = colorGen(ran);
    blue = colorGen(ran);
}
