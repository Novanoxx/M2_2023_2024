#include <glimac/SDLWindowManager.hpp>
#include <GL/glew.h>
#include <iostream>

#include <glimac/Sphere.hpp>
#include <glimac/common.hpp>
#include <glimac/Program.hpp>
#include <glimac/FilePath.hpp>
#include <glimac/glm.hpp>

using namespace glimac;
using namespace glm;

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
    Program program = loadProgram(  applicationPath.dirPath() + "shaders/3D.vs.glsl",
                                    applicationPath.dirPath() + "shaders/normals.fs.glsl");
    program.use();

    auto locMVP = glGetUniformLocation(program.getGLId(), "uMVPMatrix");
    auto locMV = glGetUniformLocation(program.getGLId(), "uMVMatrix");
    auto locCoord = glGetUniformLocation(program.getGLId(), "uNormalMatrix");

    glEnable(GL_DEPTH_TEST);

    /*********************************
     * HERE SHOULD COME THE INITIALIZATION CODE
     *********************************/

    
    mat4 ProjMatrix = perspective(radians(70.f), largeur/hauteur, 0.1f, 100.f);
    mat4 MVMatrix = translate(mat4(1.f), vec3(0.f, 0.f, -5.f));
    mat4 NormalMatrix = transpose(inverse(MVMatrix));
    
    // VBO et VAO
    const int numSegmentsMajor = 30;
    const int numSegmentsMinor = 20;
    const float majorRadius = 2.0f;
    const float minorRadius = 0.5f;
    GLuint vbo, vao, ibo;
    GLfloat* vertices = new GLfloat[3 * (numSegmentsMajor + 1) * (numSegmentsMinor + 1)];
    GLuint* indices = new GLuint[6 * numSegmentsMajor * numSegmentsMinor];

    int vertexIndex = 0;
    int index = 0;

    for (int i = 0; i <= numSegmentsMajor; ++i) {
        for (int j = 0; j <= numSegmentsMinor; ++j) {
            float thetaMajor = 2.0f * 3.14159265359f * i / numSegmentsMajor;
            float phiMinor = 2.0f * 3.14159265359f * j / numSegmentsMinor;

            float x = (majorRadius + minorRadius * std::cos(phiMinor)) * std::cos(thetaMajor);
            float y = (majorRadius + minorRadius * std::cos(phiMinor)) * std::sin(thetaMajor);
            float z = minorRadius * std::sin(phiMinor);

            vertices[vertexIndex++] = x;
            vertices[vertexIndex++] = y;
            vertices[vertexIndex++] = z;

            if (i < numSegmentsMajor && j < numSegmentsMinor) {
                indices[index++] = i * (numSegmentsMinor + 1) + j;
                indices[index++] = (i + 1) * (numSegmentsMinor + 1) + j;
                indices[index++] = i * (numSegmentsMinor + 1) + j + 1;

                indices[index++] = (i + 1) * (numSegmentsMinor + 1) + j;
                indices[index++] = (i + 1) * (numSegmentsMinor + 1) + j + 1;
                indices[index++] = i * (numSegmentsMinor + 1) + j + 1;
            }
        }
    }

    glGenBuffers(1, &vbo);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, 3 * (numSegmentsMajor + 1) * (numSegmentsMinor + 1) * sizeof(GLfloat), vertices, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glGenBuffers(1, &ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 6 * numSegmentsMajor * numSegmentsMinor * sizeof(GLuint), indices, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);

    const GLuint VERTEX_ATTR_POSITION = 0;
    glEnableVertexAttribArray(VERTEX_ATTR_POSITION);
    glVertexAttribPointer(VERTEX_ATTR_POSITION, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), nullptr);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindVertexArray(0);

    std::cout << "END" << std::endl;

    // END INITIALIZATION

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
        glUniformMatrix4fv(locMVP, 1, GL_FALSE, value_ptr(ProjMatrix * MVMatrix));
        glUniformMatrix4fv(locMV, 1, GL_FALSE, value_ptr(MVMatrix));
        glUniformMatrix4fv(locCoord, 1, GL_FALSE, value_ptr(NormalMatrix));

        glBindVertexArray(vao);
        //glDrawArrays(GL_TRIANGLES, 0, 6 * numSegmentsMajor * numSegmentsMinor);
        glDrawElements(GL_TRIANGLES, 6 * numSegmentsMajor * numSegmentsMinor, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);

        // END RENDERING

        // Update the display
        windowManager.swapBuffers();
    }
    delete[] vertices;
    delete[] indices;
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);

    return EXIT_SUCCESS;
}
