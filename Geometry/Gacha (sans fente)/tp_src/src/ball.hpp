#ifndef ball_hpp
#define ball_hpp

#include <c3ga/Mvec.hpp>
#include "c3gaTools.hpp"
#include "Geogebra_c3ga.hpp"
#include "Entry.hpp"
#include "ball.hpp"
#include <random>

using namespace std;

typedef uniform_real_distribution<> urd;

class Ball
{
public:
    Ball();
    Ball(const Ball& a);
    c3ga::Mvec<float> centre;
    c3ga::Mvec<float> velocity;
    float red;
    float green;
    float blue;
    float radius = 0.3;
    float mass = radius * radius * radius;
    
    Ball& operator=(const Ball& a);
    
    void initBall(urd& xposGen, urd& yposGen,urd& velGen, urd& colorGen, mt19937& ran);
};

#endif
