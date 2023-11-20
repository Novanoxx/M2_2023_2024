#include <glimac/SDLWindowManager.hpp>
#include <GL/glew.h>
#include <iostream>

#include <glimac/Program.hpp>
#include <glimac/FilePath.hpp>
#include <glimac/glm.hpp>

using namespace glimac;

struct Vertex2DColor
{
    glm::vec2 position;
    glm::vec3 color;

    Vertex2DColor()
    {
        // Default constructor
    }

    Vertex2DColor(glm::vec2 pos, glm::vec3 col)
    {
        position = pos;
        color = col;
    }
};

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

    /*********************************
     * HERE COME THE INITIALIZATION CODE
     *********************************/
     // Creation d'un unique vbo
    GLuint vbo;
    glGenBuffers(1, &vbo);

    // Binding d'un VBO sur la cible GL_ARRAY_BUFFER (de type GLenum)
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    Vertex2DColor vertices[] = {    Vertex2DColor(glm::vec2(-0.5, -0.5), glm::vec3(1, 0, 0)),
                                    Vertex2DColor(glm::vec2(0.5, -0.5), glm::vec3(0, 1, 0)),
                                    Vertex2DColor(glm::vec2(0, 0.5), glm::vec3(0, 0, 1))
                        };
    glBufferData(GL_ARRAY_BUFFER, 3 * sizeof(Vertex2DColor), vertices, GL_STATIC_DRAW);

    // Debind le vbo
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    

    // Creation d'un unique vao
    GLuint vao;
    glGenVertexArrays(1, &vao);

    // Binding d'un VAO
    glBindVertexArray(vao);

    const GLuint VERTEX_ATTR_POSITION = 0;
    const GLuint VERTEX_ATTR_COLOR = 1;
    glEnableVertexAttribArray(VERTEX_ATTR_POSITION);
    glEnableVertexAttribArray(VERTEX_ATTR_COLOR);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(VERTEX_ATTR_POSITION, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), offsetof(Vertex2DColor, position));
    glVertexAttribPointer(VERTEX_ATTR_COLOR, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (const GLvoid*) offsetof(Vertex2DColor, color));
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Debind le vao
    glBindVertexArray(0);

    // END OF INITIALIZATION

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
         * HERE COME THE RENDERING CODE
         *********************************/
        glClear(GL_COLOR_BUFFER_BIT);
        
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glBindVertexArray(0);

        // END OF RENDERING

        // Update the display
        windowManager.swapBuffers();
    }

    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);

    return EXIT_SUCCESS;
}
