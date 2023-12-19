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
    GLuint uSecondaryexture;

    TwoLayersPlanetProgram(const FilePath& applicationPath):
        m_Program(loadProgram(applicationPath.dirPath() + "shaders/3D.vs.glsl",
                        applicationPath.dirPath() + "shaders/multiTex3D.fs.glsl"))
    {
        uMVPMatrix = glGetUniformLocation(m_Program.getGLId(), "uMVPMatrix");
        uMVMatrix = glGetUniformLocation(m_Program.getGLId(), "uMVMatrix");
        uNormalMatrix = glGetUniformLocation(m_Program.getGLId(), "uNormalMatrix");
        uMainTexture = glGetUniformLocation(m_Program.getGLId(), "uMainTexture");
        uSecondaryexture = glGetUniformLocation(m_Program.getGLId(), "uSecondaryTexture");
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

void updateUniform(PlanetProgram star, mat4 MVMatrix, mat4 NormalMatrix, mat4 ProjMatrix)
{
    glUniformMatrix4fv(star.uMVMatrix, 1, GL_FALSE, value_ptr(MVMatrix));
    glUniformMatrix4fv(star.uNormalMatrix, 1, GL_FALSE, value_ptr(NormalMatrix));
    glUniformMatrix4fv(star.uMVPMatrix, 1, GL_FALSE, value_ptr(ProjMatrix * MVMatrix));
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

int main(int argc, char** argv) {
    // Initialize SDL and open a window
    float largeur = 1000.f;
    float hauteur = 800.f;
    SDLWindowManager windowManager(largeur, hauteur, "GLImac");

    // Initialize glew for OpenGL3+ support
    GLenum glewInitError = glewInit();
    if(GLEW_OK != glewInitError) {
        std::cerr << glewGetErrorString(glewInitError) << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "OpenGL Version : " << glGetString(GL_VERSION) << std::endl;
    std::cout << "GLEW Version : " << glewGetString(GLEW_VERSION) << std::endl;

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

    // END OF LOADING TEXTURE

    glEnable(GL_DEPTH_TEST);

    /*********************************
     * HERE SHOULD COME THE INITIALIZATION CODE
     *********************************/

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
    TrackballCamera tracking;

    while(!done) {
        // Init matrix
        mat4 ProjMatrix = perspective(radians(70.f), largeur/hauteur, 0.1f, 1000.f);
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
                    case 122:          // z key 
                        tracking.moveFront(1);
                        break;
                    case 273 :         // up key
                        tracking.moveFront(1);
                        break;
                    case 115 :          // s key 
                        tracking.moveFront(-1);
                        break;
                    case 274 :          // down key
                        tracking.moveFront(-1);
                        break;
                    case 112 :          // p key
                        pause = !pause;
                        break;
                    default :
                        break;
                }
                std::cout << e.key.keysym.sym << std::endl;
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

            // Sun
            float sunSpeed = windowManager.getTime()/50;
            sunProgram.m_Program.use();
            glUniform1i(sunProgram.uTexture, 0);
            mat4 sunMVMatrix = rotate(globalMVMatrix, sunSpeed, vec3(0, 1, 0));
            updateUniform(sunProgram, sunMVMatrix, NormalMatrix, ProjMatrix);

            drawOneLayer(textures[0], sphere);
            sleep(0.5);

            /**** Mercury
            distance from sun : 57.9 --> /20 --> 2,9
            diameter : 4879 --> 0,49
            orbital velocity : 47.4 km/s
            orbital angle : 9°
            ****/
            mercuryProgram.m_Program.use();
            glUniform1i(mercuryProgram.uMainTexture, 0);
            glUniform1i(mercuryProgram.uSecondaryexture, 1);
            mat4 mercuryMVMatrix = RTS(sunMVMatrix, sunSpeed*47.4f, -2.9f, 9.f, 0.2f, 0.2f, 0.2f);
            // mat4 mercuryMVMatrix = RTS(sunMVMatrix, sunSpeed*47.4f, -2.9f, 9.f, 0.049f, 0.049f, 0.049f);
            updateUniform(mercuryProgram, mercuryMVMatrix, NormalMatrix, ProjMatrix);
            
            drawTwoLayers(textures[1], textures[2], sphere);
            sleep(0.5);

            /**** Venus
            distance from sun : 108.2 --> 5,4
            diameter : 12,104 --> 1,21
            orbital velocity : 35.0 km/s
            orbital angle : 3.4°
            ****/
            venusProgram.m_Program.use();
            glUniform1i(venusProgram.uMainTexture, 0);
            glUniform1i(venusProgram.uSecondaryexture, 1);
            mat4 venusMVMatrix = RTS(sunMVMatrix, sunSpeed*35.f, -5.4f, 3.4f, 0.4f, 0.4f, 0.4f);
            // mat4 venusMVMatrix = RTS(sunMVMatrix, sunSpeed, -5.4f, 3.4f, 0.121f, 0.121f, 0.121f);
            updateUniform(venusProgram, venusMVMatrix, NormalMatrix, ProjMatrix);
            
            drawTwoLayers(textures[3], textures[4], sphere);
            sleep(0.5);

            /**** Earth
            distance from sun : 149.6 --> 7,5
            diameter : 12,756  --> 1,28
            orbital velocity : 29.8 km/s
            orbital angle : 0° (based from Earth so 0)
            ****/
            earthProgram.m_Program.use();
            glUniform1i(earthProgram.uMainTexture, 0);
            glUniform1i(earthProgram.uSecondaryexture, 1);
            mat4 earthMVMatrix = RTS(sunMVMatrix, sunSpeed*29.8f, -7.5f, 1.f, 0.5f, 0.5f, 0.5f);
            // mat4 earthMVMatrix = RTS(sunMVMatrix, sunSpeed, -7.5f, 1.f, 0.128f, 0.128f, 0.128f);
            updateUniform(earthProgram, earthMVMatrix, NormalMatrix, ProjMatrix);
            
            drawTwoLayers(textures[5], textures[6], sphere);
            sleep(0.5);

            /**** Moon (satellite)
            distance from Earth : 0.384
            diameter : 3475 --> 0,35
            orbital velocity : 1 km/s
            orbital angle : 5.1°
            ****/
            moonProgram.m_Program.use();
            glUniform1i(moonProgram.uTexture, 0);
            mat4 MVMatrixMoon = rotate(earthMVMatrix, 23.f, vec3(0.2f, 5.1f, 0.2f));    // without this, there is no satellite
            MVMatrixMoon = RTS(MVMatrixMoon, 1.f, -2.f, 1.f, 0.2f, 0.2f, 0.2f);
            // MVMatrixMoon = RTS(MVMatrixMoon, sunSpeed, -2.f, 5.1f, 0.035f, 0.035f, 0.035f);
            updateUniform(moonProgram, MVMatrixMoon, NormalMatrix, ProjMatrix);

            drawOneLayer(textures[7], sphere);

            /**** Mars
            distance from sun : 228.0 --> 11,4
            diameter : 6792 --> 0,68
            orbital velocity : 24.1 km/s
            orbital angle : 1.8°
            ****/
            marsProgram.m_Program.use();
            glUniform1i(marsProgram.uTexture, 0);
            mat4 marsMVMatrix = RTS(sunMVMatrix, sunSpeed*24.1f, -11.4f, 1.8f, 0.3f, 0.3f, 0.3f);
            // mat4 marsMVMatrix = RTS(sunMVMatrix, sunSpeed, -11.4f, 1.8f, 0.068f, 0.068f, 0.068f);
            updateUniform(marsProgram, marsMVMatrix, NormalMatrix, ProjMatrix);

            drawOneLayer(textures[8], sphere);
            sleep(0.5);
            // Deimos (satellite)
            // Phobos (satellite)

            /**** Jupiter
            distance from sun : 778.5 --> 38,9
            diameter : 142,984 --> 14,3
            orbital velocity : 13.1 km/s
            orbital angle : 1.3°
            ****/ 
            jupiterProgram.m_Program.use();
            glUniform1i(jupiterProgram.uTexture, 0);
            mat4 jupiterMVMatrix = RTS(sunMVMatrix, sunSpeed*13.1f, -38.9f, 1.3f, 0.9f, 0.9f, 0.9f);
            // mat4 jupiterMVMatrix = RTS(sunMVMatrix, sunSpeed, -38.9f, 1.3f, 0.143f, 0.143f, 0.143f);
            updateUniform(jupiterProgram, jupiterMVMatrix, NormalMatrix, ProjMatrix);

            drawOneLayer(textures[11], sphere);
            sleep(0.5);
            /**** Saturn
            distance from sun : 1432.0 --> 71,6
            diameter : 120,536 --> 12,05
            orbital velocity : 9.7 km/s
            orbital angle : 2.5°
            ****/
            saturnProgram.m_Program.use();
            glUniform1i(saturnProgram.uTexture, 0);
            mat4 saturnMVMatrix = RTS(sunMVMatrix, sunSpeed*9.7f, -71.6f, 2.5f, 0.8f, 0.8f, 0.8f);
            // mat4 saturnMVMatrix = RTS(sunMVMatrix, sunSpeed, -71.6f, 2.5f, 0.12f, 0.12f, 0.12f);
            updateUniform(saturnProgram, saturnMVMatrix, NormalMatrix, ProjMatrix);

            drawOneLayer(textures[12], sphere);
            sleep(0.5);
            // Saturn ring

            /**** Uranus
            distance from sun : 2867.0 --> 143,4
            diameter : 51,118 --> 5,11
            orbital velocity : 6.8 km/s
            orbital angle : 0.8°
            ****/ 
            uranusProgram.m_Program.use();
            glUniform1i(uranusProgram.uTexture, 0);
            mat4 uranusMVMatrix = RTS(sunMVMatrix, sunSpeed*6.8f, -143.4f, 0.8f, 0.7f, 0.7f, 0.7f);
            // mat4 uranusMVMatrix = RTS(sunMVMatrix, sunSpeed, -143.4f, 0.8f, 0.511f, 0.511f, 0.511f);
            updateUniform(uranusProgram, uranusMVMatrix, NormalMatrix, ProjMatrix);

            drawOneLayer(textures[14], sphere);
            sleep(0.5);
            // Uranus ring

            /**** Neptune
            distance from sun : 4515.0 --> 225,8
            diameter : 49,528 --> 4,95
            orbital velocity : 5.4 km/s
            orbital angle : 1.8°
            ****/
            neptuneProgram.m_Program.use();
            glUniform1i(neptuneProgram.uTexture, 0);
            mat4 neptuneMVMatrix = RTS(sunMVMatrix, sunSpeed*5.4f, -225.8f, 1.8f, 0.6f, 0.6f, 0.6f);
            // mat4 neptuneMVMatrix = RTS(sunMVMatrix, sunSpeed, -225.8f, 1.8f, 0.495f, 0.495f, 0.495f);
            updateUniform(neptuneProgram, neptuneMVMatrix, NormalMatrix, ProjMatrix);

            drawOneLayer(textures[16], sphere);
            sleep(0.5);

            /**** Pluto
            distance from sun : 5906.4 --> 295,3
            diameter : 2376 --> 0,24
            orbital velocity : 4.7 km/s
            orbital angle : 17.2°
            ****/
            plutoProgram.m_Program.use();
            glUniform1i(plutoProgram.uMainTexture, 0);
            glUniform1i(plutoProgram.uSecondaryexture, 1);
            mat4 plutoMVMatrix = RTS(sunMVMatrix, sunSpeed*4.7f, -295.3f, 17.2f, 0.1f, 0.1f, 0.1f);
            // mat4 plutoMVMatrix = RTS(sunMVMatrix, sunSpeed, -295.3f, 17.2f, 0.024f, 0.024f, 0.024f);
            updateUniform(plutoProgram, plutoMVMatrix, NormalMatrix, ProjMatrix);
            
            drawTwoLayers(textures[17], textures[18], sphere);
            sleep(0.5);

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
