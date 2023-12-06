#include <glimac/SDLWindowManager.hpp>
#include <GL/glew.h>
#include <iostream>

#include <glimac/Sphere.hpp>
#include <glimac/common.hpp>
#include <glimac/Program.hpp>
#include <glimac/FilePath.hpp>
#include <glimac/glm.hpp>
#include <glimac/Image.hpp>

using namespace glimac;
using namespace glm;

void loadTexture(GLuint *textureArray, int idTexture, std::string image) {
    glBindTexture(GL_TEXTURE_2D, textureArray[idTexture]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    auto texture = loadImage("../assets/textures/" + image + ".jpg");
    if (texture) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texture->getWidth(), texture->getHeight(), 0, GL_RGBA, GL_FLOAT, texture->getPixels());
        std::cout << "Loading" << image << "texture" << std::endl;
    } else {
        std::cout << "Failed to load " << image << " texture" << std::endl;
    }
    glBindTexture(GL_TEXTURE_2D, 0);
}

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

    // Load and bind texture
    int nbTextures = 3;
    GLuint textures[nbTextures];
    glGenTextures(nbTextures, textures);
    loadTexture(textures, 0, "EarthMap");
    loadTexture(textures, 1, "CloudMap");
    loadTexture(textures, 2, "MoonMap");

    // END OF LOADING TEXTURE

    std::cout << "OpenGL Version : " << glGetString(GL_VERSION) << std::endl;
    std::cout << "GLEW Version : " << glewGetString(GLEW_VERSION) << std::endl;

    FilePath applicationPath(argv[0]);
    Program program = loadProgram(  applicationPath.dirPath() + "shaders/" + argv[1],
                                    applicationPath.dirPath() + "shaders/" + argv[2]);
    program.use();

    auto locMVP = glGetUniformLocation(program.getGLId(), "uMVPMatrix");
    auto locMV = glGetUniformLocation(program.getGLId(), "uMVMatrix");
    auto locCoord = glGetUniformLocation(program.getGLId(), "uNormalMatrix");
    auto locEarthTexture = glGetUniformLocation(program.getGLId(), "uMainTexture");
    auto locMoonTexture = glGetUniformLocation(program.getGLId(), "uSecondaryTexture");

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
    mat4 ProjMatrix = perspective(radians(70.f), largeur/hauteur, 0.1f, 100.f);
    mat4 MVMatrix = translate(mat4(1.f), vec3(0.f, 0.f, -5.f));
    auto MVOrigin = MVMatrix;
    mat4 MVMatrixMoon;
    mat4 NormalMatrix = transpose(inverse(MVMatrix));

    vec3 randArray[32];
    for (int i = 0; i < 32; i++)
    {
        randArray[i] = sphericalRand(1.f);
    }

    // Application loop:
    bool done = false;
    while(!done) {
        // Event loop:
        SDL_Event e;
        while(windowManager.pollEvent(e)) {
            if(e.type == SDL_QUIT) {
                done = true; // Leave the loop after this iteration
            }
        }

        /*********************************
         * HERE SHOULD COME THE RENDERING CODE
         *********************************/

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glClearColor(0.2f, 0.2f, 0.2f, 1.f);
        MVMatrix = rotate(MVMatrix, 0.01f, vec3(0, 1, 0));
        glUniformMatrix4fv(locMVP, 1, GL_FALSE, value_ptr(ProjMatrix * MVMatrix));
        glUniformMatrix4fv(locMV, 1, GL_FALSE, value_ptr(MVMatrix));
        glUniformMatrix4fv(locCoord, 1, GL_FALSE, value_ptr(NormalMatrix));
        glUniform1i(locEarthTexture, 0);
        glUniform1i(locEarthTexture, 1);
        glUniform1i(locMoonTexture, 0);


        glBindVertexArray(vao);

        // Earth
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, textures[0]);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, textures[1]);
        glDrawArrays(GL_TRIANGLES, 0, sphere.getVertexCount());
        glBindTexture(GL_TEXTURE_2D, 0);

        // Moons
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, textures[2]);
        for (int i = 0; i < 32; i++)
        {   
            MVMatrixMoon = rotate(MVOrigin, 23.f, randArray[i]);
            MVMatrixMoon = rotate(MVMatrixMoon, windowManager.getTime(), vec3(0, 1, 0));
            MVMatrixMoon = translate(MVMatrixMoon, vec3(-2, 0, 0));
            MVMatrixMoon = scale(MVMatrixMoon, vec3(0.2, 0.2, 0.2));
            glUniformMatrix4fv(locMVP, 1, GL_FALSE, value_ptr(ProjMatrix * MVMatrixMoon));
            glUniformMatrix4fv(locMV, 1, GL_FALSE, value_ptr(MVMatrixMoon));
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
    glDeleteTextures(1, textures);

    return EXIT_SUCCESS;
}
