#version 330

in vec3 vFragPosition;
in vec3 vFragNormals;
in vec2 vFragTexCoords;

out vec3 fFragColor;

uniform sampler2D uEarthTexture;
uniform sampler2D uMoonTexture;

void main() {
    vec4 textureA = texture(uEarthTexture, vFragTexCoords);
    vec4 textureB = texture(uMoonTexture, vFragTexCoords);
    fFragColor = normalize(vFragNormals);
    fFragColor = textureA.xyz + textureB.xyz;

}