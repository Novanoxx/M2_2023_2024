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

void updatePlanetUniform(PlanetProgram star, mat4 MVMatrix, mat4 NormalMatrix, mat4 ProjMatrix)
{
    glUniformMatrix4fv(star.uMVMatrix, 1, GL_FALSE, value_ptr(MVMatrix));
    glUniformMatrix4fv(star.uNormalMatrix, 1, GL_FALSE, value_ptr(NormalMatrix));
    glUniformMatrix4fv(star.uMVPMatrix, 1, GL_FALSE, value_ptr(ProjMatrix * MVMatrix));
}

void updateSatelliteUniform(PlanetProgram star, mat4 MVMatrix, mat4 ProjMatrix)
{
    glUniformMatrix4fv(star.uMVPMatrix, 1, GL_FALSE, value_ptr(ProjMatrix * MVMatrix));
    glUniformMatrix4fv(star.uMVMatrix, 1, GL_FALSE, value_ptr(MVMatrix));
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

    // Satellite
    OneLayerPlanetProgram moonProgram(applicationPath);

    // Load and bind texture
    int nbTextures = 8;
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

    /*
    // Mars
    loadTexture(textures, 8, "Mars/mars_1k_color");
    loadTexture(textures, 9, "Mars/deimosbump");
    loadTexture(textures, 10, "Mars/phobosbump");

    // Jupiter
    loadTexture(textures, 11, "Jupiter/jupiter2_1k");
    loadTexture(textures, 12, "Jupiter/jupitermap");

    // Saturn
    loadTexture(textures, 13, "Saturn/saturnmap");
    loadTexture(textures, 14, "Saturn/saturnringcolor");

    // Uranus
    loadTexture(textures, 15, "Uranus/uranusmap");
    loadTexture(textures, 16, "Uranus/uranusringcoulour");

    // Neptune
    loadTexture(textures, 17, "Neptune/neptunemap");

    // Pluto
    loadTexture(textures, 18, "Pluto/plutobump1k");
    loadTexture(textures, 19, "Pluto/plutomap1k");
    */

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
                    default :
                        break;
                }
                std::cout << e.key.keysym.sym << std::endl;
            }
        }

        /*********************************
         * HERE SHOULD COME THE RENDERING CODE
         *********************************/
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(0.2f, 0.2f, 0.2f, 1.f);
        
        glBindVertexArray(vao);

        sleep(0.5);
        // Sun
        sunProgram.m_Program.use();
        glUniform1i(sunProgram.uTexture, 0);
        mat4 sunMVMatrix = rotate(globalMVMatrix, windowManager.getTime()/2, vec3(0, 1, 0));
        updatePlanetUniform(sunProgram, sunMVMatrix, NormalMatrix, ProjMatrix);

        drawOneLayer(textures[0], sphere);
        sleep(0.5);

        // Mercury
        mercuryProgram.m_Program.use();
        glUniform1i(mercuryProgram.uMainTexture, 0);
        glUniform1i(mercuryProgram.uSecondaryexture, 1);
        mat4 mercuryMVMatrix = rotate(sunMVMatrix, windowManager.getTime()/2, vec3(0, 1, 0));
        mercuryMVMatrix = translate(mercuryMVMatrix, vec3(-2, 0, 0));
        mercuryMVMatrix = scale(mercuryMVMatrix, vec3(0.3, 0.3, 0.3));
        updatePlanetUniform(mercuryProgram, mercuryMVMatrix, NormalMatrix, ProjMatrix);
        
        drawTwoLayers(textures[1], textures[2], sphere);
        sleep(0.5);

        // Venus
        venusProgram.m_Program.use();
        glUniform1i(venusProgram.uMainTexture, 0);
        glUniform1i(venusProgram.uSecondaryexture, 1);
        mat4 venusMVMatrix = rotate(sunMVMatrix, windowManager.getTime()/2, vec3(0, 1, 0));
        venusMVMatrix = translate(venusMVMatrix, vec3(-4, 0, 0));
        venusMVMatrix = scale(venusMVMatrix, vec3(0.6, 0.6, 0.6));
        updatePlanetUniform(venusProgram, venusMVMatrix, NormalMatrix, ProjMatrix);
        
        drawTwoLayers(textures[3], textures[4], sphere);
        sleep(0.5);

        // Earth
        earthProgram.m_Program.use();
        glUniform1i(earthProgram.uMainTexture, 0);
        glUniform1i(earthProgram.uSecondaryexture, 1);
        mat4 earthMVMatrix = rotate(sunMVMatrix, windowManager.getTime()/2, vec3(0, 1, 0));
        earthMVMatrix = translate(earthMVMatrix, vec3(-6, 0, 0));
        earthMVMatrix = scale(earthMVMatrix, vec3(0.5, 0.5, 0.5));
        updatePlanetUniform(earthProgram, earthMVMatrix, NormalMatrix, ProjMatrix);
        
        drawTwoLayers(textures[5], textures[6], sphere);
        sleep(0.5);

        // Moon (satellite)
        moonProgram.m_Program.use();
        glUniform1i(moonProgram.uTexture, 0);
        mat4 MVMatrixMoon = rotate(earthMVMatrix, 23.f, vec3(0.2, 0.2, 0.2));
        MVMatrixMoon = rotate(MVMatrixMoon, windowManager.getTime(), vec3(0, 1, 0));
        MVMatrixMoon = translate(MVMatrixMoon, vec3(-2, 0, 0));
        MVMatrixMoon = scale(MVMatrixMoon, vec3(0.2, 0.2, 0.2));
        updateSatelliteUniform(moonProgram, MVMatrixMoon, ProjMatrix);

        drawOneLayer(textures[7], sphere);

        // Mars
        // Deimos (satellite)
        // Phobos (satellite)
        // Jupiter
        // Saturn
        // Saturn ring
        // Uranus
        // Uranus ring
        // Neptune

        glBindVertexArray(0);
        // END RENDERING

        // Update the display
        windowManager.swapBuffers();
    }
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    //glDeleteTextures(1, textures);

    return EXIT_SUCCESS;
}
