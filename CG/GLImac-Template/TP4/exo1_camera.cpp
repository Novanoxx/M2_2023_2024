#include <glimac/SDLWindowManager.hpp>
#include <GL/glew.h>
#include <iostream>
#include <unistd.h>

#include <glimac/Sphere.hpp>
#include <glimac/common.hpp>
#include <glimac/Program.hpp>
#include <glimac/FilePath.hpp>
#include <glimac/glm.hpp>
#include <glimac/Image.hpp>
#include <glimac/TrackballCamera.hpp>

using namespace glimac;
using namespace glm;

void loadTexture(GLuint *textureArray, int idTexture, std::string image) {
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

struct EarthProgram {
    Program m_Program;
    GLint uMVPMatrix;
    GLint uMVMatrix;
    GLint uNormalMatrix;
    GLuint uEarthTexture;
    GLuint uCloudTexture;

    EarthProgram(const FilePath& applicationPath):
    m_Program(loadProgram(applicationPath.dirPath() + "shaders/3D.vs.glsl",
                        applicationPath.dirPath() + "shaders/multiTex3D.fs.glsl"))
    {
        uMVPMatrix = glGetUniformLocation(m_Program.getGLId(), "uMVPMatrix");
        uMVMatrix = glGetUniformLocation(m_Program.getGLId(), "uMVMatrix");
        uNormalMatrix = glGetUniformLocation(m_Program.getGLId(), "uNormalMatrix");
        uEarthTexture = glGetUniformLocation(m_Program.getGLId(), "uMainTexture");
        uCloudTexture = glGetUniformLocation(m_Program.getGLId(), "uSecondaryTexture");
    }
};

struct MoonProgram {
    Program m_Program;

    GLint uMVPMatrix;
    GLint uMVMatrix;
    GLint uNormalMatrix;
    GLuint uTexture;

    MoonProgram(const FilePath& applicationPath):
        m_Program(loadProgram(applicationPath.dirPath() + "shaders/3D.vs.glsl",
                              applicationPath.dirPath() + "shaders/tex3D.fs.glsl"))
    {
        uMVPMatrix = glGetUniformLocation(m_Program.getGLId(), "uMVPMatrix");
        uMVMatrix = glGetUniformLocation(m_Program.getGLId(), "uMVMatrix");
        uNormalMatrix = glGetUniformLocation(m_Program.getGLId(), "uNormalMatrix");
        uTexture = glGetUniformLocation(m_Program.getGLId(), "uTexture");
    }
};

int main(int argc, char** argv) {
    // Initialize SDL and open a window
    float largeur = 800.f;
    float hauteur = 600.f;
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
    EarthProgram earthProgram(applicationPath);
    MoonProgram moonProgram(applicationPath);

    // Load and bind texture
    int nbTextures = 3;
    GLuint textures[nbTextures];
    glGenTextures(nbTextures, textures);
    loadTexture(textures, 0, "EarthMap");
    loadTexture(textures, 1, "CloudMap");
    loadTexture(textures, 2, "MoonMap");
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

    vec3 randArray[32];
    for (int i = 0; i < 32; i++)
    {
        randArray[i] = sphericalRand(1.f);
    }
    
    mat4 ProjMatrix;
    mat4 globalMVMatrix;
    mat4 NormalMatrix;
    mat4 MVMatrixMoon;
    mat4 earthMVMatrix;

    TrackballCamera tracking;

    // Application loop:
    bool done = false;
    while(!done) {
        ProjMatrix = perspective(radians(70.f), largeur/hauteur, 0.1f, 10000.f);
        globalMVMatrix = translate(mat4(1.f), vec3(0.f, 0.f, -5.f));
        NormalMatrix = transpose(inverse(globalMVMatrix));
        
        // Event loop:
        SDL_Event e;
        int xMouse, yMouse;

        //ProjMatrix += tracking.getViewMatrix();
        globalMVMatrix *= tracking.getViewMatrix();

        while(windowManager.pollEvent(e)) {
            if(e.type == SDL_MOUSEMOTION)
            {
                if(e.motion.state & SDL_BUTTON_LMASK)
                {
                    SDL_GetMouseState(&xMouse, &yMouse);
                    //tracking.rotateLeft(10);
                    tracking.rotateLeft((xMouse - hauteur/2)/100);
                    tracking.rotateUp((yMouse - largeur/2)/100);
                    //std::cout << "x : " << xMouse << " , y : " << yMouse << std::endl;
                }
            }
            if(e.type == SDL_QUIT)
            {
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
        //glClearColor(0.2f, 0.2f, 0.2f, 1.f);
        
        glBindVertexArray(vao);

        // Earth
        earthProgram.m_Program.use();
        glUniform1i(earthProgram.uEarthTexture, 0);
        glUniform1i(earthProgram.uCloudTexture, 1);
        earthMVMatrix = rotate(globalMVMatrix, windowManager.getTime(), vec3(0, 1, 0));
        glUniformMatrix4fv(earthProgram.uMVMatrix, 1, GL_FALSE, value_ptr(earthMVMatrix));
        glUniformMatrix4fv(earthProgram.uNormalMatrix, 1, GL_FALSE, value_ptr(NormalMatrix));
        glUniformMatrix4fv(earthProgram.uMVPMatrix, 1, GL_FALSE, value_ptr(ProjMatrix * earthMVMatrix));
        
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, textures[0]);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, textures[1]);
        glDrawArrays(GL_TRIANGLES, 0, sphere.getVertexCount());
        glBindTexture(GL_TEXTURE_2D, 0);

        // Moons
        moonProgram.m_Program.use();
        glUniform1i(moonProgram.uTexture, 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, textures[2]);
        for (int i = 0; i < 32; i++)
        {    
            MVMatrixMoon = rotate(globalMVMatrix, 23.f, randArray[i]);
            MVMatrixMoon = rotate(MVMatrixMoon, windowManager.getTime(), vec3(0, 1, 0));
            MVMatrixMoon = translate(MVMatrixMoon, vec3(-2, 0, 0));
            MVMatrixMoon = scale(MVMatrixMoon, vec3(0.2, 0.2, 0.2));
            glUniformMatrix4fv(moonProgram.uMVPMatrix, 1, GL_FALSE, value_ptr(ProjMatrix * MVMatrixMoon));
            glUniformMatrix4fv(moonProgram.uMVMatrix, 1, GL_FALSE, value_ptr(MVMatrixMoon));
            glDrawArrays(GL_TRIANGLES, 0, sphere.getVertexCount());
        }
        glBindTexture(GL_TEXTURE_2D, 0);

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
