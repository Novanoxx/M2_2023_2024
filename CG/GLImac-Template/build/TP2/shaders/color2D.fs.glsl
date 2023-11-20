#version 330 core

in vec3 vFragColor;
in vec2 vFragPosition;

out vec3 fFragColor;

void main() {
  float a = 3.0f;
  float b = 100.0f;
  float dist = distance(vFragPosition, vec2(0.f, 0.f));
  float A = a * exp(-b * pow(dist, 2));

  fFragColor = vFragColor * A;
};