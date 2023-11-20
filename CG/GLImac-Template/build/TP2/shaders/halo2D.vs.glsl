#version 330 core

layout(location = 0) in vec2 aVertexPosition;
layout(location = 1) in vec3 aVertexColor;

out vec3 vFragColor;
out vec2 vFragPosition;

mat3 translate(float tx, float ty) {
  return mat3(vec3(1, 0, 0),
              vec3(0, 1, 0),
              vec3(tx, ty, 1));
}

mat3 scale(float sx, float sy) {
  return mat3(vec3(sx, 0, 0),
              vec3(0, sy, 0),
              vec3(0, 0, 1));
}

mat3 rotate(float a) {
  return mat3(vec3(cos(a), sin(a), 0),
              vec3(-sin(a), cos(a), 0),
              vec3(0, 0, 1));
}

void main() {
  vFragColor = aVertexColor;
  vFragPosition = aVertexPosition;
  mat3 T = translate(0.5f, 0.0f);
  //mat3 R = rotate(radians(45));
  //mat3 S = scale(0.5f, 0.5f);
  gl_Position = vec4((T * vec3(aVertexPosition, 1)).xy, 0, 1);
};