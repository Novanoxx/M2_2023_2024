#version 330 core

layout(location = 0) in vec2 aVertexPosition;
layout(location = 1) in vec3 aVertexColor;

out vec3 vFragColor;

void main() {
  vFragColor = aVertexColor;
  gl_Position = vec4(aVertexPosition.x * 2.0f, aVertexPosition.y / 2.0f, 0, 1);
};