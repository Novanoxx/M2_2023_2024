#include <glimac/SDLWindowManager.hpp>
#include <GL/glew.h>
#include <iostream>

#include <glimac/Sphere.hpp>
#include <glimac/common.hpp>
#include <glimac/Program.hpp>
#include <glimac/FilePath.hpp>
#include <glimac/glm.hpp>
#include <glimac/Image.hpp>
#include <glimac/TrackballCamera.hpp>
#include <unistd.h>

using namespace glimac;
using namespace glm;

struct PlanetProgram
{
    GLint uMVPMatrix;
    GLint uMVMatrix;
    GLint uNormalMatrix;
};

struct TwoLayersPlanetProgram : PlanetProgram
{
    Program m_Program;

    GLuint uMainTexture;
    GLuint uSecondaryTexture;

    TwoLayersPlanetProgram(const FilePath& applicationPath):
        m_Program(loadProgram(applicationPath.dirPath() + "shaders/3D.vs.glsl",
                        applicationPath.dirPath() + "shaders/multiTex3D.fs.glsl"))
    {
        uMVPMatrix = glGetUniformLocation(m_Program.getGLId(), "uMVPMatrix");
        uMVMatrix = glGetUniformLocation(m_Program.getGLId(), "uMVMatrix");
        uNormalMatrix = glGetUniformLocation(m_Program.getGLId(), "uNormalMatrix");
        uMainTexture = glGetUniformLocation(m_Program.getGLId(), "uMainTexture");
        uSecondaryTexture = glGetUniformLocation(m_Program.getGLId(), "uSecondaryTexture");
    }
};

struct OneLayerPlanetProgram : PlanetProgram
{
    Program m_Program;

    GLuint uTexture;

    OneLayerPlanetProgram(const FilePath& applicationPath):
        m_Program(loadProgram(applicationPath.dirPath() + "shaders/3D.vs.glsl",
                              applicationPath.dirPath() + "shaders/tex3D.fs.glsl"))
    {
        uMVPMatrix = glGetUniformLocation(m_Program.getGLId(), "uMVPMatrix");
        uMVMatrix = glGetUniformLocation(m_Program.getGLId(), "uMVMatrix");
        uNormalMatrix = glGetUniformLocation(m_Program.getGLId(), "uNormalMatrix");
        uTexture = glGetUniformLocation(m_Program.getGLId(), "uTexture");
    }
};

void loadTexture(GLuint *textureArray, int idTexture, std::string image)
{
    glBindTexture(GL_TEXTURE_2D, textureArray[idTexture]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    auto texture = loadImage("../assets/textures/" + image + ".jpg");
    if (texture) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texture->getWidth(), texture->getHeight(), 0, GL_RGBA, GL_FLOAT, texture->getPixels());
        std::cout << "Loading " << image << " texture" << std::endl;
    } else {
        std::cout << "Failed to load " << image << " texture" << std::endl;
    }
    glBindTexture(GL_TEXTURE_2D, 0);
}

void updateUniform(PlanetProgram planet, mat4 MVMatrix, mat4 NormalMatrix, mat4 ProjMatrix)
{
    glUniformMatrix4fv(planet.uMVMatrix, 1, GL_FALSE, value_ptr(MVMatrix));
    glUniformMatrix4fv(planet.uNormalMatrix, 1, GL_FALSE, value_ptr(NormalMatrix));
    glUniformMatrix4fv(planet.uMVPMatrix, 1, GL_FALSE, value_ptr(ProjMatrix * MVMatrix));
}

void drawOneLayer(GLuint texture, Sphere sphere)
{
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);
    glDrawArrays(GL_TRIANGLES, 0, sphere.getVertexCount());
    glBindTexture(GL_TEXTURE_2D, 0);
}

void drawTwoLayers(GLuint textureA, GLuint textureB, Sphere sphere)
{
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, textureA);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, textureB);
    glDrawArrays(GL_TRIANGLES, 0, sphere.getVertexCount());
    glBindTexture(GL_TEXTURE_2D, 0);
}

mat4 RTS(mat4 origin, float time, float tX, float angle, float sX, float sY, float sZ)
{
    mat4 ret = rotate(origin, 23.f, vec3(0.f, angle, 0.f));
    ret = rotate(ret, time, vec3(0, 1, 0));
    ret = translate(ret, vec3(tX, 0, 0));
    ret = scale(ret, vec3(sX, sY, sZ));
    return ret;
}

void loadAllTextures(int nbTextures, GLuint *textures)
{
    glGenTextures(nbTextures, textures);

    // Sun
    loadTexture(textures, 0, "Sun/sunmap");
    // Mercury
    loadTexture(textures, 1, "Mercury/mercurybump");
    loadTexture(textures, 2, "Mercury/mercurymap");
    // Venus
    loadTexture(textures, 3, "Venus/venusbump");
    loadTexture(textures, 4, "Venus/venusmap");
    // Earth
    loadTexture(textures, 5, "Earth/EarthMap");
    loadTexture(textures, 6, "Earth/CloudMap");
    loadTexture(textures, 7, "Earth/MoonMap");
    // Mars
    loadTexture(textures, 8, "Mars/mars_1k_color");
    loadTexture(textures, 9, "Mars/deimosbump");
    loadTexture(textures, 10, "Mars/phobosbump");
    // Jupiter
    loadTexture(textures, 11, "Jupiter/jupiter2_1k");
    // Saturn
    loadTexture(textures, 12, "Saturn/saturnmap");
    loadTexture(textures, 13, "Saturn/saturnringcolor");
    // Uranus
    loadTexture(textures, 14, "Uranus/uranusmap");
    loadTexture(textures, 15, "Uranus/uranusringcolour");
    // Neptune
    loadTexture(textures, 16, "Neptune/neptunemap");
    // Pluto
    loadTexture(textures, 17, "Pluto/plutobump1k");
    loadTexture(textures, 18, "Pluto/plutomap1k");
}

mat4 drawOLP(OneLayerPlanetProgram &planet, mat4 origin, mat4 NormalMatrix, mat4 ProjMatrix, float time, float tX, float angle, float sX, float sY, float sZ, GLuint texture, Sphere sphere)
{
    planet.m_Program.use();
    glUniform1i(planet.uTexture, 0);
    mat4 MVMatrix = RTS(origin, time, tX, angle, sX, sY, sZ);
    updateUniform(planet, MVMatrix, NormalMatrix, ProjMatrix);
    
    drawOneLayer(texture, sphere);
    sleep(0.5);

    return MVMatrix;
}

mat4 drawTLP(TwoLayersPlanetProgram &planet, mat4 origin, mat4 NormalMatrix, mat4 ProjMatrix, float time, float tX, float angle, float sX, float sY, float sZ, GLuint textureA, GLuint textureB, Sphere sphere)
{
    planet.m_Program.use();
    glUniform1i(planet.uMainTexture, 0);
    glUniform1i(planet.uSecondaryTexture, 1);
    mat4 MVMatrix = RTS(origin, time, tX, angle, sX, sY, sZ);
    updateUniform(planet, MVMatrix, NormalMatrix, ProjMatrix);
    
    drawTwoLayers(textureA, textureB, sphere);
    sleep(0.5);

    return MVMatrix;
}

mat4 drawOLPZoom(OneLayerPlanetProgram &planet, mat4 origin, mat4 NormalMatrix, mat4 ProjMatrix, float time, GLuint texture, Sphere sphere)
{
    planet.m_Program.use();
    glUniform1i(planet.uTexture, 0);
    mat4 MVMatrix = rotate(origin, time, vec3(0, 1, 0));
    updateUniform(planet, MVMatrix, NormalMatrix, ProjMatrix);

    drawOneLayer(texture, sphere);
    sleep(0.5);

    return MVMatrix;
}

mat4 drawTLPZoom(TwoLayersPlanetProgram &planet, mat4 origin, mat4 NormalMatrix, mat4 ProjMatrix, float time, GLuint textureA, GLuint textureB, Sphere sphere)
{
    planet.m_Program.use();
    glUniform1i(planet.uMainTexture, 0);
    glUniform1i(planet.uSecondaryTexture, 1);
    
    mat4 MVMatrix = rotate(origin, time, vec3(0, 1, 0));
    updateUniform(planet, MVMatrix, NormalMatrix, ProjMatrix);

    drawTwoLayers(textureA, textureB, sphere);
    sleep(0.5);

    return MVMatrix;
}

void showPlanet(std::vector<bool> &planets, int index, bool &all)
{
    for (int i = 0; i < planets.size(); i++)
    {
        planets.at(i) = false;
    }
    planets.at(index) = true;
    all = false;
}

void cameraZoom(TrackballCamera &camera)
{
    camera.reset();
    camera.rotateUp(-65.f);
    camera.moveFront(10.f);
}

int main(int argc, char** argv) {
    // Initialize SDL and open a window
    float width = 1000.f;
    float height = 800.f;
    SDLWindowManager windowManager(width, height, "GLImac");

    // Initialize glew for OpenGL3+ support
    GLenum glewInitError = glewInit();
    if(GLEW_OK != glewInitError) {
        std::cerr << glewGetErrorString(glewInitError) << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "OpenGL Version : " << glGetString(GL_VERSION) << std::endl;
    std::cout << "GLEW Version : " << glewGetString(GLEW_VERSION) << std::endl;

    glEnable(GL_DEPTH_TEST);

    /*********************************
     * HERE SHOULD COME THE INITIALIZATION CODE
     *********************************/
    FilePath applicationPath(argv[0]);

    // Planet
    OneLayerPlanetProgram sunProgram(applicationPath);
    TwoLayersPlanetProgram mercuryProgram(applicationPath);
    TwoLayersPlanetProgram venusProgram(applicationPath);
    TwoLayersPlanetProgram earthProgram(applicationPath);
    OneLayerPlanetProgram marsProgram(applicationPath);
    OneLayerPlanetProgram jupiterProgram(applicationPath);
    OneLayerPlanetProgram saturnProgram(applicationPath);
    OneLayerPlanetProgram uranusProgram(applicationPath);
    OneLayerPlanetProgram neptuneProgram(applicationPath);
    TwoLayersPlanetProgram plutoProgram(applicationPath);

    // Satellite
    OneLayerPlanetProgram moonProgram(applicationPath);

    // Load and bind texture
    int nbTextures = 19;
    GLuint textures[nbTextures];
    loadAllTextures(nbTextures, textures);
    // END OF LOADING TEXTURE

    Sphere sphere(1, 32, 16);
    
    // VBO et VAO
    GLuint vbo;
    glGenBuffers(1, &vbo);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sphere.getVertexCount() * sizeof(ShapeVertex), sphere.getDataPointer(), GL_STATIC_DRAW);

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    const GLuint VERTEX_ATTR_POSITION = 0;
    const GLuint VERTEX_ATTR_NORMAL = 1;
    const GLuint VERTEX_ATTR_COORD = 2;

    glEnableVertexAttribArray(VERTEX_ATTR_POSITION);
    glEnableVertexAttribArray(VERTEX_ATTR_NORMAL);
    glEnableVertexAttribArray(VERTEX_ATTR_COORD);

    glVertexAttribPointer(VERTEX_ATTR_POSITION, 3, GL_FLOAT, GL_FALSE, sizeof(ShapeVertex), offsetof(ShapeVertex, position));
    glVertexAttribPointer(VERTEX_ATTR_NORMAL, 3, GL_FLOAT, GL_FALSE, sizeof(ShapeVertex), (const GLvoid*) offsetof(ShapeVertex, normal));
    glVertexAttribPointer(VERTEX_ATTR_COORD, 2, GL_FLOAT, GL_FALSE, sizeof(ShapeVertex), (const GLvoid*) offsetof(ShapeVertex, texCoords));
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindVertexArray(0);
    // END INITIALIZATION

    // Application loop:
    bool done = false;
    bool pause = false;
    bool all = true;
    // bool sun, mercury, venus, earth, march, jupiter, saturn, uranus, neptune, pluto = false;
    std::vector<bool> planets = {false, false, false, false, false, false, false, false, false, false};
    
    TrackballCamera tracking;
    float lastX = 0.0f;
    float lastY = 0.0f;

    while(!done) {
        // Init matrix
        mat4 ProjMatrix = perspective(radians(70.f), width/height, 0.1f, 1000.f);
        mat4 globalMVMatrix = translate(mat4(1.f), vec3(0.f, 0.f, -5.f));
        mat4 NormalMatrix = transpose(inverse(globalMVMatrix));

        // Event loop:
        SDL_Event e;
        int xMouse, yMouse;
        
        globalMVMatrix *= tracking.getViewMatrix();
        while(windowManager.pollEvent(e)) {
            if(e.type == SDL_QUIT) {
                done = true; // Leave the loop after this iteration
            }
            if (e.type == SDL_KEYDOWN)
            {
                switch(e.key.keysym.sym)
                {
                    case 273 :         // up key
                        tracking.moveFront(1);
                        break;
                    case 274 :          // down key
                        tracking.moveFront(-1);
                        break;
                    case 112 :          // p key
                        pause = !pause;
                        break;
                    case 119 :          // w key for all
                        all = true;
                        for (int i = 0; i < planets.size(); i++)
                        {
                            planets.at(i) = false;
                        }
                        tracking.reset();
                        break;
                    case 113 :          // q key for the Sun
                        cameraZoom(tracking);
                        showPlanet(planets, 0, all);
                        break;
                    case 115 :          // s key for Mercury
                        cameraZoom(tracking);
                        showPlanet(planets, 1, all);
                        break;
                    case 100 :          // d key for Venus
                        cameraZoom(tracking);
                        showPlanet(planets, 2, all);
                        break;
                    case 102 :          // f key for Earth
                        cameraZoom(tracking);
                        showPlanet(planets, 3, all);
                        break;
                    case 103 :          // g key for March
                        cameraZoom(tracking);
                        showPlanet(planets, 4, all);
                        break;
                    case 104 :          // h key for Jupiter
                        cameraZoom(tracking);
                        showPlanet(planets, 5, all);
                        break;
                    case 106 :          // j key for Saturn
                        cameraZoom(tracking);
                        showPlanet(planets, 6, all);
                        break;
                    case 107 :          // k key for Uranus
                        cameraZoom(tracking);
                        showPlanet(planets, 7, all);
                        break;
                    case 108 :          // l key for Neptune
                        cameraZoom(tracking);
                        showPlanet(planets, 8, all);
                        break;
                    case 109 :          // m key for Pluto
                        cameraZoom(tracking);
                        showPlanet(planets, 9, all);
                        break;
                    default :
                        break;
                }
                //std::cout << e.key.keysym.sym << std::endl;
            }
            if (e.type == SDL_MOUSEMOTION)
            {
                if(e.motion.state & SDL_BUTTON_LMASK)
                {
                    int xpos, ypos;
                    xpos = e.motion.x;
                    ypos = e.motion.y;

                    float xOffset = xpos - lastX;
                    float yOffset = lastY - ypos;

                    if (xOffset != 0)
                    {
                        xOffset > 0 ? tracking.rotateLeft(cos(radians(xOffset))) : tracking.rotateLeft(-cos(radians(xOffset)));
                    }
                    if (yOffset != 0)
                    {
                        yOffset > 0 ? tracking.rotateUp(-cos(radians(yOffset))) : tracking.rotateUp(cos(radians(yOffset)));
                    }

                    lastX = xpos;
                    lastY = ypos;
                }
            }
        }

        /*********************************
         * HERE SHOULD COME THE RENDERING CODE
         *********************************/
        if (!pause)
        {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glClearColor(0.2f, 0.2f, 0.2f, 1.f);
            
            glBindVertexArray(vao);

            sleep(0.5);
            /***
            Size : 
            Jupiter, Saturn, Uranus, Nepturne, Earth, Venus, Mars, Mercury, Pluto
            ***/
            if (all)
            {
                float sunSpeed = windowManager.getTime()/50;
                mat4 sunMVMatrix = drawOLPZoom(sunProgram, globalMVMatrix, NormalMatrix, ProjMatrix, sunSpeed, textures[0], sphere);

                drawTLP(mercuryProgram, sunMVMatrix, NormalMatrix, ProjMatrix, sunSpeed * 47.4f,
                                    -2.9f, 9.f, 0.2f, 0.2f, 0.2f, textures[1], textures[2], sphere);

                drawTLP(venusProgram, sunMVMatrix, NormalMatrix, ProjMatrix, sunSpeed * 35.f,
                                    -5.4f, 3.4f, 0.4f, 0.4f, 0.4f, textures[3], textures[4], sphere);
                
                mat4 earthMVMatrix = drawTLP(earthProgram, sunMVMatrix, NormalMatrix, ProjMatrix, sunSpeed * 29.8f,
                                    -7.5f, 1.f, 0.5f, 0.5f, 0.5f, textures[5], textures[6], sphere);
                
                mat4 MVMatrixMoon = rotate(earthMVMatrix, 23.f, vec3(0.2f, 5.1f, 0.2f));    // without this, there is no satellite
                drawOLP(moonProgram, MVMatrixMoon, NormalMatrix, ProjMatrix, 1.f,
                                    -2.f, 1.f, 0.2f, 0.2f, 0.2f, textures[7], sphere);

                drawOLP(marsProgram, sunMVMatrix, NormalMatrix, ProjMatrix, sunSpeed * 24.1f,
                                    -11.4f, 1.8f, 0.3f, 0.3f, 0.3f, textures[8], sphere);

                drawOLP(jupiterProgram, sunMVMatrix, NormalMatrix, ProjMatrix, sunSpeed * 13.1f,
                                    -38.9f, 1.3f, 0.9f, 0.9f, 0.9f, textures[11], sphere);                                    

                drawOLP(saturnProgram, sunMVMatrix, NormalMatrix, ProjMatrix, sunSpeed * 9.7f,
                                    -71.6f, 2.5f, 0.8f, 0.8f, 0.8f, textures[12], sphere); 

                drawOLP(uranusProgram, sunMVMatrix, NormalMatrix, ProjMatrix, sunSpeed * 6.8f,
                                    -143.4f, 0.8f, 0.7f, 0.7f, 0.7f, textures[14], sphere);  

                drawOLP(neptuneProgram, sunMVMatrix, NormalMatrix, ProjMatrix, sunSpeed * 5.4f,
                                    -225.8f, 1.8f, 0.6f, 0.6f, 0.6f, textures[14], sphere);                                      

                drawTLP(plutoProgram, sunMVMatrix, NormalMatrix, ProjMatrix, sunSpeed * 4.7f,
                                    -295.3f, 17.2f, 0.1f, 0.1f, 0.1f, textures[17], textures[18], sphere);
            } else
            {
                if (planets.at(0))
                {
                    drawOLPZoom(sunProgram, globalMVMatrix, NormalMatrix, ProjMatrix, windowManager.getTime(), textures[0], sphere);
                }
                if (planets.at(1))
                {
                    drawTLPZoom(mercuryProgram, globalMVMatrix, NormalMatrix, ProjMatrix, windowManager.getTime(), textures[1], textures[2], sphere);
                }
                if (planets.at(2))
                {
                    drawTLPZoom(venusProgram, globalMVMatrix, NormalMatrix, ProjMatrix, windowManager.getTime(), textures[3], textures[4], sphere);
                }
                if (planets.at(3))
                {
                    mat4 earthMVMatrix = drawTLPZoom(earthProgram, globalMVMatrix, NormalMatrix, ProjMatrix, windowManager.getTime(), textures[5], textures[6], sphere);
                    mat4 MVMatrixMoon = rotate(earthMVMatrix, 23.f, vec3(0.2f, 5.1f, 0.2f));    // without this, there is no satellite
                    drawOLP(moonProgram, MVMatrixMoon, NormalMatrix, ProjMatrix, windowManager.getTime(),
                                        -2.f, 1.f, 0.2f, 0.2f, 0.2f, textures[7], sphere);
                }
                if (planets.at(4))
                {
                    drawOLPZoom(marsProgram, globalMVMatrix, NormalMatrix, ProjMatrix, windowManager.getTime(), textures[8], sphere);
                }
                if (planets.at(5))
                {
                    drawOLPZoom(jupiterProgram, globalMVMatrix, NormalMatrix, ProjMatrix, windowManager.getTime(), textures[11], sphere);
                }
                if (planets.at(6))
                {
                    drawOLPZoom(saturnProgram, globalMVMatrix, NormalMatrix, ProjMatrix, windowManager.getTime(), textures[12], sphere);
                }
                if (planets.at(7))
                {
                    drawOLPZoom(uranusProgram, globalMVMatrix, NormalMatrix, ProjMatrix, windowManager.getTime(), textures[14], sphere);
                }
                if (planets.at(8))
                {
                    drawOLPZoom(neptuneProgram, globalMVMatrix, NormalMatrix, ProjMatrix, windowManager.getTime(), textures[16], sphere);
                }
                if (planets.at(9))
                {
                    drawTLPZoom(plutoProgram, globalMVMatrix, NormalMatrix, ProjMatrix, windowManager.getTime(), textures[17], textures[18], sphere);
                }
            }
            // Sun

            /**** Mercury
            distance from sun : 57.9 --> /20 --> 2,9
            diameter : 4879 --> 0,49
            orbital velocity : 47.4 km/s
            orbital angle : 9°
            // mat4 mercuryMVMatrix = RTS(sunMVMatrix, sunSpeed * 47.4f, -2.9f, 9.f, 0.049f, 0.049f, 0.049f);
            ****/

            /**** Venus
            distance from sun : 108.2 --> 5,4
            diameter : 12,104 --> 1,21
            orbital velocity : 35.0 km/s
            orbital angle : 3.4°
            // mat4 venusMVMatrix = RTS(sunMVMatrix, sunSpeed * 35.f, -5.4f, 3.4f, 0.121f, 0.121f, 0.121f);
            ****/
            

            /**** Earth
            distance from sun : 149.6 --> 7,5
            diameter : 12,756  --> 1,28
            orbital velocity : 29.8 km/s
            orbital angle : 0° (based from Earth so 0)
            // mat4 earthMVMatrix = RTS(sunMVMatrix, sunSpeed, -7.5f, 1.f, 0.128f, 0.128f, 0.128f);
            ****/
            

            /**** Moon (satellite)
            distance from Earth : 0.384
            diameter : 3475 --> 0,35
            orbital velocity : 1 km/s
            orbital angle : 5.1°
            ****/
            

            /**** Mars
            distance from sun : 228.0 --> 11,4
            diameter : 6792 --> 0,68
            orbital velocity : 24.1 km/s
            orbital angle : 1.8°
            ****/
            
            // Deimos (satellite)
            // Phobos (satellite)

            /**** Jupiter
            distance from sun : 778.5 --> 38,9
            diameter : 142,984 --> 14,3
            orbital velocity : 13.1 km/s
            orbital angle : 1.3°
            ****/ 

            /**** Saturn
            distance from sun : 1432.0 --> 71,6
            diameter : 120,536 --> 12,05
            orbital velocity : 9.7 km/s
            orbital angle : 2.5°
            ****/

            // Saturn ring

            /**** Uranus
            distance from sun : 2867.0 --> 143,4
            diameter : 51,118 --> 5,11
            orbital velocity : 6.8 km/s
            orbital angle : 0.8°
            ****/ 

            // Uranus ring

            /**** Neptune
            distance from sun : 4515.0 --> 225,8
            diameter : 49,528 --> 4,95
            orbital velocity : 5.4 km/s
            orbital angle : 1.8°
            ****/


            /**** Pluto
            distance from sun : 5906.4 --> 295,3
            diameter : 2376 --> 0,24
            orbital velocity : 4.7 km/s
            orbital angle : 17.2°
            ****/
            

            glBindVertexArray(0);
            // END RENDERING

            // Update the display
            windowManager.swapBuffers();
        }
    }
        
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    //glDeleteTextures(1, textures);

    return EXIT_SUCCESS;
}
