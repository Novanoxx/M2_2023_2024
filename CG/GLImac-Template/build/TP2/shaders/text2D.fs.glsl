#version 330 core

in vec3 vFragColor;
in vec2 vFragPosition;

out vec3 fFragColor;

void main() {
  float dist = distance(vFragPosition, vec2(0.f, 0.f));
  //float A = length(fract(5.0f * dist));
  float A = smoothstep(0.4, 0.5, max(abs(fract(8.0 * vFragPosition.x - 0.5 * mod(floor(8.0 * vFragPosition.y), 2.0)) - 0.5), abs(fract(8.0 * vFragPosition.y) - 0.5)));

  fFragColor = vFragColor * A;
};