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

mat4 RTS(mat4 origin, float selfRotation, float orbitRotation, float tX, float e, float angle, float sX, float sY, float sZ)
{
    mat4 ret = rotate(origin, orbitRotation, vec3(e, std::sin(angle), radians(angle)));         // Orbit rotation
    ret = translate(ret, vec3(tX, 0.f, 0.f));                                                   // Distance of from X
    ret = rotate(ret, selfRotation, vec3(0.f, 1.f, 0.f));                                       // Self rotation
    ret = scale(ret, vec3(sX, sY, sZ));
    return ret;
}

mat4 drawOLP(OneLayerPlanetProgram &planet, mat4 origin, mat4 NormalMatrix, mat4 ProjMatrix, float selfRotation, float orbitRotation, 
                float tX, float angle, float e, float sX, float sY, float sZ, GLuint texture, Sphere sphere)
{
    planet.m_Program.use();
    glUniform1i(planet.uTexture, 0);
    mat4 MVMatrix = RTS(origin, selfRotation, orbitRotation, tX, e, angle, sX, sY, sZ);
    updateUniform(planet, MVMatrix, NormalMatrix, ProjMatrix);
    
    drawOneLayer(texture, sphere);
    sleep(0.5);

    return MVMatrix;
}

mat4 drawTLP(TwoLayersPlanetProgram &planet, mat4 origin, mat4 NormalMatrix, mat4 ProjMatrix, float selfRotation, float orbitRotation,
                float tX, float angle, float e, float sX, float sY, float sZ, GLuint textureA, GLuint textureB, Sphere sphere)
{
    planet.m_Program.use();
    glUniform1i(planet.uMainTexture, 0);
    glUniform1i(planet.uSecondaryTexture, 1);
    mat4 MVMatrix = RTS(origin, selfRotation, orbitRotation, tX, e, angle, sX, sY, sZ);
    updateUniform(planet, MVMatrix, NormalMatrix, ProjMatrix);
    
    drawTwoLayers(textureA, textureB, sphere);
    sleep(0.5);

    return MVMatrix;
}

mat4 drawOLPZoom(OneLayerPlanetProgram &planet, mat4 origin, mat4 NormalMatrix, mat4 ProjMatrix, float selfRotation, float angle, GLuint texture, Sphere sphere)
{
    planet.m_Program.use();
    glUniform1i(planet.uTexture, 0);
    
    mat4 MVMatrix = rotate(origin, selfRotation, vec3(0, 1.f, 0));
    updateUniform(planet, MVMatrix, NormalMatrix, ProjMatrix);

    drawOneLayer(texture, sphere);
    sleep(0.5);

    return MVMatrix;
}

mat4 drawTLPZoom(TwoLayersPlanetProgram &planet, mat4 origin, mat4 NormalMatrix, mat4 ProjMatrix, float selfRotation, float angle, GLuint textureA, GLuint textureB, Sphere sphere)
{
    planet.m_Program.use();
    glUniform1i(planet.uMainTexture, 0);
    glUniform1i(planet.uSecondaryTexture, 1);
    
    mat4 MVMatrix = rotate(origin, selfRotation, vec3(0, 1.f, 0));
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

    // Planets
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

    // Satellites
    // Moon
    OneLayerPlanetProgram moonProgram(applicationPath);
    // Deimos, Phobos
    OneLayerPlanetProgram deimosProgram(applicationPath);
    OneLayerPlanetProgram phobosProgram(applicationPath);
    // Callisto, Ganymède, Europa, Io    
    OneLayerPlanetProgram callistoProgram(applicationPath);
    OneLayerPlanetProgram ganymedeProgram(applicationPath);
    OneLayerPlanetProgram europaProgram(applicationPath);
    OneLayerPlanetProgram ioProgram(applicationPath);
    // Mimas, Enceladus, Tethys, Dione, Rhea, Titan, Hyperion, Iapetus + anneaux
    OneLayerPlanetProgram mimasProgram(applicationPath);
    OneLayerPlanetProgram enceladusProgram(applicationPath);
    OneLayerPlanetProgram tethysProgram(applicationPath);
    OneLayerPlanetProgram dioneProgram(applicationPath);
    OneLayerPlanetProgram rheaProgram(applicationPath);
    OneLayerPlanetProgram titanProgram(applicationPath);
    OneLayerPlanetProgram hyperionProgram(applicationPath);
    OneLayerPlanetProgram iapetusProgram(applicationPath);
    // Ariel, Umbriel, Titania, Obéron, Miranda + anneaux
    OneLayerPlanetProgram arielProgram(applicationPath);
    OneLayerPlanetProgram umbrielProgram(applicationPath);
    OneLayerPlanetProgram titaniaProgram(applicationPath);
    OneLayerPlanetProgram oberonProgram(applicationPath);
    OneLayerPlanetProgram mirandaProgram(applicationPath);
    // Triton, Nereid
    OneLayerPlanetProgram tritonProgram(applicationPath);
    OneLayerPlanetProgram nereidProgram(applicationPath);
    // Charon
    OneLayerPlanetProgram charonProgram(applicationPath);

    // Load and bind texture and images
    int nbTextures = 20;
    GLuint textures[nbTextures];
    loadAllTextures(nbTextures, textures);

    // END OF LOADING TEXTURE AND IMAGES

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

    /*********************************
     * HERE SHOULD END THE INITIALIZATION CODE
     *********************************/

    // Application loop:
    bool done = false;
    bool pause = false;
    bool all = true;
    // sun, mercury, venus, earth, march, jupiter, saturn, uranus, neptune, pluto;
    std::vector<bool> planets = {false, false, false, false, false, false, false, false, false, false};
    
    TrackballCamera tracking;
    float lastX = 0.0f;
    float lastY = 0.0f;

    float speed = 1.f;

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
                    case 275 :          // right key
                        speed ++;
                        break;
                    case 276 :          // left key
                        speed --;
                        break;
                    case 32 :           // space key
                        done = true;
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
                    case 120 :          // x key for all
                        all = true;
                        for (int i = 0; i < planets.size(); i++)
                        {
                            planets.at(i) = false;
                        }
                        tracking.reset();
                        tracking.rotateUp(-65.f);
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
                // std::cout << e.key.keysym.sym << std::endl;
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

            glBindVertexArray(vao);
            sleep(0.5);
            /***
            Size : 
            Jupiter > Saturn > Uranus > Nepturne > Earth > Venus > Mars > Mercury > Pluto
            ***/
            speed <= 0 ? speed = 1.f : speed;
            auto time = (windowManager.getTime()/100) * speed;
            if (all)
            {
                mat4 sunMVMatrix = drawOLPZoom(sunProgram, globalMVMatrix, NormalMatrix, ProjMatrix, time * 0.6f, 1.f, textures[0], sphere);

                drawTLP(mercuryProgram, sunMVMatrix, NormalMatrix, ProjMatrix, time * 0.26f, time * 7.59f,
                                    -2.9f, 7.f, 0.206f, 0.2f, 0.2f, 0.2f, textures[1], textures[2], sphere);

                drawTLP(venusProgram, sunMVMatrix, NormalMatrix, ProjMatrix, time * -0.06f, time * 10.29f,
                                    -5.4f, 3.4f, 0.007f, 0.4f, 0.4f, 0.4f, textures[3], textures[4], sphere);
                
                mat4 earthMVMatrix = drawTLP(earthProgram, sunMVMatrix, NormalMatrix, ProjMatrix, time * 15.06f, time * 12.08f,
                                    -7.5f, 1.f, 0.017f, 0.5f, 0.5f, 0.5f, textures[5], textures[6], sphere);
                drawOLP(moonProgram, earthMVMatrix, NormalMatrix, ProjMatrix, time * 0.55f, time,
                                    -2.f, 1.f, 0.055f, 0.2f, 0.2f, 0.2f, textures[7], sphere);

                mat4 marsMVMatrix = drawOLP(marsProgram, sunMVMatrix, NormalMatrix, ProjMatrix, time * 14.63f, time * 14.94f,
                                    -11.4f, 1.8f, 0.094f, 0.3f, 0.3f, 0.3f, textures[8], sphere);
                drawOLP(deimosProgram, marsMVMatrix, NormalMatrix, ProjMatrix, time * 11.88f, time * 11.88f,
                                        4.6f, 1.79f, 0.0005f, 0.2f, 0.2f, 0.2f, textures[9], sphere);
                drawOLP(phobosProgram, marsMVMatrix, NormalMatrix, ProjMatrix, time * 1128.84f, time * 1128.84f,
                                    1.88f, 1.08f, 0.0151f, 0.05f, 0.05f, 0.05f, textures[10], sphere); 

                mat4 jupiterMVMatrix = drawOLP(jupiterProgram, sunMVMatrix, NormalMatrix, ProjMatrix, time * 36.36f, time * 27.48f,
                                    -38.9f, 1.3f, 0.049f, 0.9f, 0.9f, 0.9f, textures[11], sphere);                                    
                drawOLP(callistoProgram, jupiterMVMatrix, NormalMatrix, ProjMatrix, time * 0.9f, time * 0.9f,
                                    18.82f, 0.19f, 0.007f, 0.2410f, 0.2410f, 0.2410f, textures[10], sphere);    
                drawOLP(ganymedeProgram, jupiterMVMatrix, NormalMatrix, ProjMatrix, time * 2.1f, time * 2.1f,
                                    10.70f, 0.18f, 0.001f, 0.2631f, 0.2631f, 0.2631f, textures[10], sphere);    
                drawOLP(europaProgram, jupiterMVMatrix, NormalMatrix, ProjMatrix, time * 4.22f, time * 4.22f,
                                    6.71f, 0.47f, 0.009f, 0.1560f, 0.1560f, 0.1560f, textures[10], sphere);    
                drawOLP(ioProgram, jupiterMVMatrix, NormalMatrix, ProjMatrix, time * 8.48f, time * 8.48f,
                                    4.21f, 0.04f, 0.004f, 0.1821f, 0.1821f, 0.1821f, textures[10], sphere);  

                mat4 saturnMVMatrix = drawOLP(saturnProgram, sunMVMatrix, NormalMatrix, ProjMatrix, time * 33.64f, time * 37.11f,
                                    -71.6f, 2.5f, 0.052f, 0.8f, 0.8f, 0.8f, textures[12], sphere); 
                drawOLP(mimasProgram, saturnMVMatrix, NormalMatrix, ProjMatrix, time * 15.92f, time * 15.92f,
                                    1.86f, 1.53f, 0.0202f, 0.0208f, 0.0197f, 0.0191f, textures[10], sphere);    
                drawOLP(enceladusProgram, saturnMVMatrix, NormalMatrix, ProjMatrix, time * 10.95f, time * 10.95f,
                                    2.38f, 0.f, 0.0045f, 0.0257f, 0.0251f, 0.0248f, textures[10], sphere);    
                drawOLP(tethysProgram, saturnMVMatrix, NormalMatrix, ProjMatrix, time * 7.94f, time * 7.94f,
                                    2.95f, 1.86f, 0.f, 0.0538f, 0.0528f, 0.0526f, textures[10], sphere);    
                drawOLP(dioneProgram, saturnMVMatrix, NormalMatrix, ProjMatrix, time * 5.48f, time * 5.48f,
                                    3.77f, 0.02f, 0.0022f, 0.0563f, 0.0561f, 0.0560f, textures[10], sphere);    
                drawOLP(rheaProgram, saturnMVMatrix, NormalMatrix, ProjMatrix, time * 3.32f, time * 3.32f,
                                    5.27f, 0.35f, 0.0010f, 0.0765f, 0.0763f, 0.0762f, textures[10], sphere);    
                drawOLP(titanProgram, saturnMVMatrix, NormalMatrix, ProjMatrix, time * 0.94f, time * 0.94f,
                                    12.22f, 0.33f, 0.0292f, 0.2575f, 0.2575f, 0.2575f, textures[10], sphere);    
                drawOLP(hyperionProgram, saturnMVMatrix, NormalMatrix, ProjMatrix, time * 0.7f, time * (rand()%31 - 15),
                                    15.01f, 0.43f, 0.1042f, 0.0180f, 0.0133f, 0.0103f, textures[10], sphere);    
                drawOLP(iapetusProgram, saturnMVMatrix, NormalMatrix, ProjMatrix, time * 0.2f, time * 0.2f,
                                    35.61f, 14.72f, 0.0283f, 0.0746f, 0.0746f, 0.0712f, textures[10], sphere);  

                mat4 uranusMVMatrix = drawOLP(uranusProgram, sunMVMatrix, NormalMatrix, ProjMatrix, time * -20.93f, time * 52.94f,
                                    -143.4f, 0.8f, 0.047f, 0.7f, 0.7f, 0.7f, textures[14], sphere);  
                drawOLP(mirandaProgram, uranusMVMatrix, NormalMatrix, ProjMatrix, time * 10.61f, time * 10.61f,
                                        1.3f, 4.34f, 0.0013f, 0.0240f, 0.0234f, 0.0233f, textures[10], sphere);
                drawOLP(arielProgram, uranusMVMatrix, NormalMatrix, ProjMatrix, time * 5.95f, time * 5.95f,
                                    1.91f, 0.04f, 0.0012f, 0.0581f, 0.0578f, 0.0578f, textures[10], sphere);    
                drawOLP(umbrielProgram, uranusMVMatrix, NormalMatrix, ProjMatrix, time * 3.62f, time * 3.62f,
                                    2.66f, 0.13f, 0.0039f, 0.0585f, 0.0585f, 0.0585f, textures[10], sphere);    
                drawOLP(titaniaProgram, uranusMVMatrix, NormalMatrix, ProjMatrix, time * 1.72f, time * 1.72f,
                                    4.36f, 0.08f, 0.0011f, 0.0789f, 0.0789f, 0.0789f, textures[10], sphere);    
                drawOLP(oberonProgram, uranusMVMatrix, NormalMatrix, ProjMatrix, time * 1.11f, time * 1.11f,
                                    5.84f, 0.07f, 0.0014f, 0.0761f, 0.0761f, 0.0761f, textures[10], sphere);  

                mat4 neptuneMVMatrix = drawOLP(neptuneProgram, sunMVMatrix, NormalMatrix, ProjMatrix, time * 22.36f, time * 66.67f,
                                    -225.8f, 1.8f, 0.010f, 0.6f, 0.6f, 0.6f, textures[14], sphere);                                      
                drawOLP(tritonProgram, neptuneMVMatrix, NormalMatrix, ProjMatrix, time * 2.55f, time * 2.55f,
                                    3.55f, 157.345f, 0.000016f, 0.1353f, 0.1353f, 0.1353f, textures[10], sphere);
                drawOLP(nereidProgram, neptuneMVMatrix, NormalMatrix, ProjMatrix, time * 0.04f, 0.f,
                                    55.13f, 7.23f, 0.7512f, 0.0170f, 0.0170f, 0.0170f, textures[10], sphere);

                mat4 plutoMVMatrix = drawTLP(plutoProgram, sunMVMatrix, NormalMatrix, ProjMatrix, time * -2.35f, time * 76.59f,
                                    -295.3f, 17.2f, 0.244f, 0.1f, 0.1f, 0.1f, textures[17], textures[18], sphere);
                drawOLP(charonProgram, plutoMVMatrix, NormalMatrix, ProjMatrix, time * 2.35f, time * 2.35f,
                    1.9596f, 0.00005f, 0.f, 0.0606f, 0.0606f, 0.0606f, textures[10], sphere);
            } else
            {
                if (planets.at(0))
                {
                    drawOLPZoom(sunProgram, globalMVMatrix, NormalMatrix, ProjMatrix, time * 0.6f, 1.f, textures[0], sphere);
                }
                if (planets.at(1))
                {
                    drawTLPZoom(mercuryProgram, globalMVMatrix, NormalMatrix, ProjMatrix, time * 0.26f, 0.034f, textures[1], textures[2], sphere);
                }
                if (planets.at(2))
                {
                    drawTLPZoom(venusProgram, globalMVMatrix, NormalMatrix, ProjMatrix, time * -0.06f, 177.4f, textures[3], textures[4], sphere);
                }
                if (planets.at(3))
                {
                    mat4 earthMVMatrix = drawTLPZoom(earthProgram, globalMVMatrix, NormalMatrix, ProjMatrix, time * 15.06f, 23.4f, textures[5], textures[6], sphere);
                    drawOLP(moonProgram, earthMVMatrix, NormalMatrix, ProjMatrix, time * 0.55f, time,
                                        3.84f, 1.f, 0.055f, 0.35f, 0.35f, 0.35f, textures[7], sphere);
                }
                if (planets.at(4))
                {
                    mat4 marsMVMatrix = drawOLPZoom(marsProgram, globalMVMatrix, NormalMatrix, ProjMatrix, time * 14.63f, 25.2f, textures[8], sphere);
                    // True data of March
                    // drawOLP(deimosProgram, marsMVMatrix, NormalMatrix, ProjMatrix, time, time,
                    //                     2.3f, 1.f, 0.00225f, 0.00225f, 0.00225f, textures[9], sphere);
                    // drawOLP(phobosProgram, marsMVMatrix, NormalMatrix, ProjMatrix, time, time,
                    //                     0.94f, 1.f, 0.00124f, 0.00124f, 0.00124f, textures[10], sphere);      
                    drawOLP(deimosProgram, marsMVMatrix, NormalMatrix, ProjMatrix, time * 11.88f, time * 11.88f,
                                        4.6f, 1.79f, 0.0005f, 0.2f, 0.2f, 0.2f, textures[9], sphere);
                    drawOLP(phobosProgram, marsMVMatrix, NormalMatrix, ProjMatrix, time * 1128.84f, time * 1128.84f,
                                        1.88f, 1.08f, 0.0151f, 0.1f, 0.1f, 0.1f, textures[10], sphere);    
                }
                if (planets.at(5))
                {
                    mat4 jupiterMVMatrix = drawOLPZoom(jupiterProgram, globalMVMatrix, NormalMatrix, ProjMatrix, time * 36.36f, 3.1f, textures[11], sphere);
                    drawOLP(callistoProgram, jupiterMVMatrix, NormalMatrix, ProjMatrix, time * 0.9f, time * 0.9f,
                                        18.82f, 0.19f, 0.007f, 0.2410f, 0.2410f, 0.2410f, textures[10], sphere);    
                    drawOLP(ganymedeProgram, jupiterMVMatrix, NormalMatrix, ProjMatrix, time * 2.1f, time * 2.1f,
                                        10.70f, 0.18f, 0.001f, 0.2631f, 0.2631f, 0.2631f, textures[10], sphere);    
                    drawOLP(europaProgram, jupiterMVMatrix, NormalMatrix, ProjMatrix, time * 4.22f, time * 4.22f,
                                        6.71f, 0.47f, 0.009f, 0.1560f, 0.1560f, 0.1560f, textures[10], sphere);    
                    drawOLP(ioProgram, jupiterMVMatrix, NormalMatrix, ProjMatrix, time * 8.48f, time * 8.48f,
                                        4.21f, 0.04f, 0.004f, 0.1821f, 0.1821f, 0.1821f, textures[10], sphere);    
                }
                if (planets.at(6))
                {
                    mat4 saturnMVMatrix = drawOLPZoom(saturnProgram, globalMVMatrix, NormalMatrix, ProjMatrix, time * 33.64f, 26.7f, textures[12], sphere);
                    drawOLP(mimasProgram, saturnMVMatrix, NormalMatrix, ProjMatrix, time * 15.92f, time * 15.92f,
                                        1.86f, 1.53f, 0.0202f, 0.0208f, 0.0197f, 0.0191f, textures[10], sphere);    
                    drawOLP(enceladusProgram, saturnMVMatrix, NormalMatrix, ProjMatrix, time * 10.95f, time * 10.95f,
                                        2.38f, 0.f, 0.0045f, 0.0257f, 0.0251f, 0.0248f, textures[10], sphere);    
                    drawOLP(tethysProgram, saturnMVMatrix, NormalMatrix, ProjMatrix, time * 7.94f, time * 7.94f,
                                        2.95f, 1.86f, 0.f, 0.0538f, 0.0528f, 0.0526f, textures[10], sphere);    
                    drawOLP(dioneProgram, saturnMVMatrix, NormalMatrix, ProjMatrix, time * 5.48f, time * 5.48f,
                                        3.77f, 0.02f, 0.0022f, 0.0563f, 0.0561f, 0.0560f, textures[10], sphere);    
                    drawOLP(rheaProgram, saturnMVMatrix, NormalMatrix, ProjMatrix, time * 3.32f, time * 3.32f,
                                        5.27f, 0.35f, 0.0010f, 0.0765f, 0.0763f, 0.0762f, textures[10], sphere);    
                    drawOLP(titanProgram, saturnMVMatrix, NormalMatrix, ProjMatrix, time * 0.94f, time * 0.94f,
                                        12.22f, 0.33f, 0.0292f, 0.2575f, 0.2575f, 0.2575f, textures[10], sphere);    
                    drawOLP(hyperionProgram, saturnMVMatrix, NormalMatrix, ProjMatrix, time * 0.7f, time * (rand()%31 - 15),
                                        15.01f, 0.43f, 0.1042f, 0.0180f, 0.0133f, 0.0103f, textures[10], sphere);    
                    drawOLP(iapetusProgram, saturnMVMatrix, NormalMatrix, ProjMatrix, time * 0.2f, time * 0.2f,
                                        35.61f, 14.72f, 0.0283f, 0.0746f, 0.0746f, 0.0712f, textures[10], sphere);    
                }
                if (planets.at(7))
                {
                    mat4 uranusMVMatrix = drawOLPZoom(uranusProgram, globalMVMatrix, NormalMatrix, ProjMatrix, time * -20.93f, 97.8f, textures[14], sphere);
                    drawOLP(mirandaProgram, uranusMVMatrix, NormalMatrix, ProjMatrix, time * 10.61f, time * 10.61f,
                                        1.3f, 4.34f, 0.0013f, 0.0240f, 0.0234f, 0.0233f, textures[10], sphere);
                    drawOLP(arielProgram, uranusMVMatrix, NormalMatrix, ProjMatrix, time * 5.95f, time * 5.95f,
                                        1.91f, 0.04f, 0.0012f, 0.0581f, 0.0578f, 0.0578f, textures[10], sphere);    
                    drawOLP(umbrielProgram, uranusMVMatrix, NormalMatrix, ProjMatrix, time * 3.62f, time * 3.62f,
                                        2.66f, 0.13f, 0.0039f, 0.0585f, 0.0585f, 0.0585f, textures[10], sphere);    
                    drawOLP(titaniaProgram, uranusMVMatrix, NormalMatrix, ProjMatrix, time * 1.72f, time * 1.72f,
                                        4.36f, 0.08f, 0.0011f, 0.0789f, 0.0789f, 0.0789f, textures[10], sphere);    
                    drawOLP(oberonProgram, uranusMVMatrix, NormalMatrix, ProjMatrix, time * 1.11f, time * 1.11f,
                                        5.84f, 0.07f, 0.0014f, 0.0761f, 0.0761f, 0.0761f, textures[10], sphere);    
                                
                }
                if (planets.at(8))
                {
                    mat4 neptuneMVMatrix = drawOLPZoom(neptuneProgram, globalMVMatrix, NormalMatrix, ProjMatrix, time * 22.36f, 17.2f, textures[16], sphere);
                    drawOLP(tritonProgram, neptuneMVMatrix, NormalMatrix, ProjMatrix, time * 2.55f, time * 2.55f,
                                        3.55f, 157.345f, 0.000016f, 0.1353f, 0.1353f, 0.1353f, textures[10], sphere);
                    drawOLP(nereidProgram, neptuneMVMatrix, NormalMatrix, ProjMatrix, time * 0.04f, 0.f,
                                        55.13f, 7.23f, 0.7512f, 0.0170f, 0.0170f, 0.0170f, textures[10], sphere);
                }
                if (planets.at(9))
                {
                    mat4 plutoMVMatrix = drawTLPZoom(plutoProgram, globalMVMatrix, NormalMatrix, ProjMatrix, time * -2.35f, 122.5f, textures[17], textures[18], sphere);
                    drawOLP(charonProgram, plutoMVMatrix, NormalMatrix, ProjMatrix, time * 2.35f, time * 2.35f,
                                        1.9596f, 0.00005f, 0.f, 0.0606f, 0.0606f, 0.0606f, textures[10], sphere);
                }
            }
            glBindVertexArray(0);
            // END RENDERING

            // Update the display
            windowManager.swapBuffers();
        }
    }
        
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteTextures(1, textures);

    return EXIT_SUCCESS;
}
