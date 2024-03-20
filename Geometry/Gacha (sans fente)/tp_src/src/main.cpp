#include <iostream>
#include <cstdlib>
#include <list>
#include <math.h>
#include <random>
#include "ball.hpp"

//Include OpenGL header files, so that we can use OpenGL
#ifdef __OPENGL__
#include <OpenGL/OpenGL.h>
#include <GLUT/glut.h>

#else
#include <GL/glut.h>
#endif

///////////////////////////////////////////////////////////////
using namespace std;

vector<Ball> Balls; // vector containing numBalls Ball objects

float radius = 0.3f;
int numBalls;
int nbSegment = 30;
c3ga::Mvec<float> gravity = c3ga::point<float>(0, -0.07f, 0);

class Area
{
    public :

    float X_circle_array[31][31];   // nbSegment + 1
    float Y_circle_array[31][31];   // nbSegment + 1
    
    void operator()()
    {
        glClearColor(0.0, 0.0, 0.0, 1.0);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glEnable(GL_DEPTH_TEST);
        randInitBalls();
    }
    
    void randInitBalls()
    {        
        //  mersenne_twister_engine
        mt19937 ran((random_device())());
        
        //  Used to generate the center coordinates of the ball(X,Y)
        //  Our box width is 2.0f * 2.0f but we will generate the balls between (1.5f-radius)*(1.5f-radius)
        uniform_real_distribution<> xposGen(-1.5f + radius, 1.5f - radius);
        uniform_real_distribution<> yposGen(-1.5f + radius, 1.5f - radius);
        uniform_real_distribution<> velGen(-0.15f, 0.15f);
        uniform_real_distribution<> colorGen(0.2, 1.0);
        
        for (int i = 0; i < numBalls; i++)
        {
            Ball newBall;
            newBall.initBall(xposGen, yposGen, velGen, colorGen, ran);
            Balls.push_back(newBall);
        }
        for (int i = 0; i <= nbSegment; i++)
        {
            float angle1 = i * 2 * M_PI / nbSegment;
            for (int j = 0; j <= nbSegment; j++)
            {
                float angle2 = j * 2 * M_PI / nbSegment;
                X_circle_array[i][j] = cos(angle1) * sin(angle2);
                Y_circle_array[i][j] = sin(angle1) * sin(angle2);
            }
        }
    }

    // Dessin de la balle
    void operator()(Ball& ball)
    {
        glColor3f(ball.red, ball.green, ball.blue);
        for (int i = 0; i < nbSegment; i++)
        {
            for (int j = 0; j < nbSegment; j++)
            {
                glBegin(GL_POLYGON);
                glVertex2f(ball.centre[c3ga::E1] + ball.radius * X_circle_array[i][j], ball.centre[c3ga::E2] + ball.radius * Y_circle_array[i][j]);
                glVertex2f(ball.centre[c3ga::E1] + ball.radius * X_circle_array[i][j + 1], ball.centre[c3ga::E2] + ball.radius * Y_circle_array[i][j + 1]);
                glVertex2f(ball.centre[c3ga::E1] + ball.radius * X_circle_array[i + 1][j + 1], ball.centre[c3ga::E2] + ball.radius * Y_circle_array[i + 1][j + 1]);
                glVertex2f(ball.centre[c3ga::E1] + ball.radius * X_circle_array[i + 1][j], ball.centre[c3ga::E2] + ball.radius * Y_circle_array[i + 1][j]);
                glEnd();
            }
        }
    }
};

Area area;

class Box
{
public:
    float boxWidth = 2.0f;
    float boxHeight = 2.0f;
};

Box box;

void wallCollision(Ball& ball)
{
    if(ball.centre[c3ga::E1] + ball.radius > box.boxWidth)
    {
        ball.centre[c3ga::E1] = box.boxWidth - ball.radius;
        ball.velocity[c3ga::E1] *= -1;
    }
    if(ball.centre[c3ga::E1] - ball.radius < -box.boxWidth)
    {
        ball.centre[c3ga::E1] = -box.boxWidth + ball.radius;
        ball.velocity[c3ga::E1] *= -1;
    }
    if(ball.centre[c3ga::E2] + ball.radius > box.boxHeight)
    {
        ball.centre[c3ga::E2] = box.boxHeight - ball.radius;
        ball.velocity[c3ga::E2] *= -1;
    }
    if(ball.centre[c3ga::E2] - ball.radius < -box.boxHeight)
    {
        ball.centre[c3ga::E2] = -box.boxHeight + ball.radius;
        ball.velocity[c3ga::E2] *= -1;
    }
}

float mod(c3ga::Mvec<float> t)
{
    return sqrt(t[c3ga::E1] * t[c3ga::E1] + t[c3ga::E2] * t[c3ga::E2] + t[c3ga::E3] * t[c3ga::E3]);
}

void normalize(c3ga::Mvec<float> &t)
{
    float tmp = mod(t);
    if(tmp == 0)
        return;
    
    t[c3ga::E1] /= tmp;
    t[c3ga::E2] /= tmp;
    t[c3ga::E3] /= tmp;
}

void ballCollision(Ball &a, Ball& b)
{

    c3ga::Mvec<float> t = b.centre - a.centre;
    if(mod(t) > ((a.radius + b.radius))) // If no collision happened, get out of the function
        return;
    normalize(t);

    // Move second ball outside of first ball's radius
    c3ga::Mvec<float> fpos = c3ga::point<float>(a.centre[c3ga::E1], a.centre[c3ga::E2], a.centre[c3ga::E3]);
    t *= (a.radius + b.radius);
    fpos += t;
    b.centre = fpos;
}

// Function called at each iteration of the render loop

void update(int i)
{
    for(int j = 0; j < numBalls; j++)
    {
        Balls[j].centre += gravity;
        wallCollision(Balls[j]);
    }
    
    // ball collision
    for(int j = 0; j < numBalls; j++)
    {
        for(int k = j + 1; k < numBalls; k++)
        {
            ballCollision(Balls[j], Balls[k]);
        }
    }

    glutPostRedisplay();
    glutTimerFunc(1, update, 0);
}

void drawScene() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    
    glTranslatef(0.0f, 0.0f, -6.4f);
    
    for(int i = 0; i < numBalls; i++) {
        area(Balls[i]);
    }

    glutSwapBuffers();
}

void idle()
{
    drawScene();
}

// Called when the window is resized
void handleResize(int w, int h)
{
    // Tell OpenGL how to convert from coordinates to pixel values
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);    // Switch to setting the camera perspective

    // Set the camera perspective
    glLoadIdentity();                                   // Reset camera
    gluPerspective(45.0,                                // camera angle
                   (double)1000 / (double)1000,         // width-to-height ratio
                   1.0,                                 // near z clipping coordinate
                   200.0);                              // far z clipping coordinate
}

void handleKeypress(unsigned char key,int x, int y)
{
    switch (key) {
        case 27: // Escape key
            exit(0); // Exit the program
    }
}

void initRendering()
{
    glEnable(GL_DEPTH_TEST);
}

int main(int argc, char** argv)
{
    int n;
    cout << "Nombre de boules ? : ";
    cin >> n;
    numBalls = n;

    // Initialize GLUT
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(1000, 1000); // Set the window size
    
    // Create the window
    glutCreateWindow("Gachapon ? non, juste des boules qui tombent et qui se cognent");

    area();
    initRendering(); //Initialize rendering

    //Set handler functions for drawing, keypresses, and window resizes
    glutDisplayFunc(drawScene);
    glutTimerFunc(16, update, 0);
    glutKeyboardFunc(handleKeypress);
    glutIdleFunc(idle);
    glutReshapeFunc(handleResize);
    
    glutMainLoop(); //Start the main loop.  glutMainLoop doesn't return.
    return 0;       //This line is never reached
}