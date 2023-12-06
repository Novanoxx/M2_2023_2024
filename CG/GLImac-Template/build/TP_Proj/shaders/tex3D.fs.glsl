#version 330

in vec3 vFragPosition;
in vec3 vFragNormals;
in vec2 vFragTexCoords;

out vec3 fFragColor;

uniform sampler2D uTexture;

void main() {
    vec4 texture = texture(uTexture, vFragTexCoords);
    fFragColor = normalize(vFragNormals);
    fFragColor = texture.xyz;

}
