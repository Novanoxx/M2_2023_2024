#include <glimac/SDLWindowManager.hpp>
#include <GL/glew.h>
#include <iostream>
#include <time.h>

#include <glimac/Program.hpp>
#include <glimac/FilePath.hpp>
#include <glimac/glm.hpp>

using namespace glimac;
using namespace glm;

struct Vertex2DUV
{
    vec2 pos;
    vec2 texture;

    Vertex2DUV()
    {
        // Default constructor
    }

    Vertex2DUV(vec2 position, vec2 text)
    {
        pos = position;
        texture = text;
    }
};

mat3 translate(float tx, float ty) {
  return mat3( vec3(1, 0, 0),
                    vec3(0, 1, 0),
                    vec3(tx, ty, 1));
}

mat3 scale(float sx, float sy) {
  return mat3( vec3(sx, 0, 0),
                    vec3(0, sy, 0),
                    vec3(0, 0, 1));
}

mat3 rotate(float a) {
  return mat3( vec3(cos(radians(a)), sin(radians(a)), 0),
                    vec3(-sin(radians(a)), cos(radians(a)), 0),
                    vec3(0, 0, 1));
}

int main(int argc, char** argv) {
    // Initialize SDL and open a window
    SDLWindowManager windowManager(800, 600, "GLImac");

    // Initialize glew for OpenGL3+ support
    GLenum glewInitError = glewInit();
    if(GLEW_OK != glewInitError) {
        std::cerr << glewGetErrorString(glewInitError) << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "OpenGL Version : " << glGetString(GL_VERSION) << std::endl;
    std::cout << "GLEW Version : " << glewGetString(GLEW_VERSION) << std::endl;

    FilePath applicationPath(argv[0]);
    Program program = loadProgram(  applicationPath.dirPath() + "shaders/" + argv[1],
                                    applicationPath.dirPath() + "shaders/" + argv[2]);
    program.use();    
    // auto location = glGetUniformLocation(program.getGLId(), "uTime");
    auto location = glGetUniformLocation(program.getGLId(), "uModelMatrix");

    /*********************************
     * HERE SHOULD COME THE INITIALIZATION CODE
     *********************************/

    GLuint vbo;
    glGenBuffers(1, &vbo);

    // Binding d'un VBO sur la cible GL_ARRAY_BUFFER (de type GLenum)
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    Vertex2DUV vertices[] = {  Vertex2DUV(glm::vec2(-1, -1), glm::vec2(0, 0)),
                            Vertex2DUV(glm::vec2(1, -1), glm::vec2(0, 0)),
                            Vertex2DUV(glm::vec2(0, 1), glm::vec2(0, 0))
                        };
    glBufferData(GL_ARRAY_BUFFER, 3 * sizeof(Vertex2DUV), vertices, GL_STATIC_DRAW);

    // Debind le vbo
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Creation d'un unique vao
    GLuint vao;
    glGenVertexArrays(1, &vao);

    // Binding d'un VAO
    glBindVertexArray(vao);

    const GLuint VERTEX_ATTR_POSITION = 0;
    const GLuint VERTEX_ATTR_TEXTURE = 1;
    glEnableVertexAttribArray(VERTEX_ATTR_POSITION);
    glEnableVertexAttribArray(VERTEX_ATTR_TEXTURE);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(VERTEX_ATTR_POSITION, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex2DUV), offsetof(Vertex2DUV, pos));
    glVertexAttribPointer(VERTEX_ATTR_TEXTURE, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex2DUV), (const GLvoid*) offsetof(Vertex2DUV, texture));
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Debind le vao
    glBindVertexArray(0);

    // END OF INITIALIZATION

    // Application loop:
    bool done = false;
    // float rotation = 0.0f;
    time_t start, current;
    time(&start);
    while(!done) {
        // Event loop:
        time(&current);
        SDL_Event e;
        while(windowManager.pollEvent(e)) {
            if(e.type == SDL_QUIT) {
                done = true; // Leave the loop after this iteration
            }
        }

        /*********************************
         * HERE SHOULD COME THE RENDERING CODE
         *********************************/

        glClear(GL_COLOR_BUFFER_BIT);
        
        glBindVertexArray(vao);
        // glUniform1f(location, rotation);
        glUniformMatrix3fv(location, 1, GL_FALSE, value_ptr(rotate((current-start)*2)));
        glDrawArrays(GL_TRIANGLES, 0, 3);
        // rotation += 1.0f;
        glBindVertexArray(0);

        // END OF RENDERING

        // Update the display
        windowManager.swapBuffers();
    }

    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);

    return EXIT_SUCCESS;
}
