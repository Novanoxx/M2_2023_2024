#version 330 core

in vec3 vFragColor;
in vec2 vFragPosition;

out vec3 fFragColor;

void main() {
  fFragColor = vec3((vFragColor.x + vFragColor.y + vFragColor.z)/3).xyz;
};