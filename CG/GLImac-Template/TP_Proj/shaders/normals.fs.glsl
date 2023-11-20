#version 330

in vec3 vFragPosition;
in vec3 vFragNormals;
in vec2 vFragTexCoords;

out vec3 fFragColor;

void main() {

    fFragColor = normalize(vFragNormals);
    // fFragColor = vec3(1, 0, 0);
}
