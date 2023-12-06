#version 330

in vec3 vFragPosition;
in vec3 vFragNormals;
in vec2 vFragTexCoords;

out vec3 fFragColor;

uniform sampler2D uMainTexture;
uniform sampler2D uSecondaryTexture;

void main() {
    vec4 textureA = texture(uMainTexture, vFragTexCoords);
    vec4 textureB = texture(uSecondaryTexture, vFragTexCoords);
    fFragColor = normalize(vFragNormals);
    fFragColor = textureA.xyz + textureB.xyz;

}