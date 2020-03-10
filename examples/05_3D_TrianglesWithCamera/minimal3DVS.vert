#version 330

layout(location=0) in vec4 inPosition;
layout(location=1) in float inDensity;

out vec4 color; // this is passed to the fragment shader

uniform float maxDensity; // has the same value for all vertices

uniform mat4 modelMat; // matrix to go from object space to world space
uniform mat4 viewMat; // matrix to go from world space to camera space
uniform mat4 projectionMat; // matrix to perform perspective projection

void main()
{
    gl_Position = projectionMat * viewMat * modelMat * vec4(inPosition.xyz,1);

    // apply transfer function to the density
    color = vec4(inDensity / maxDensity, inDensity / maxDensity, inDensity/ maxDensity, 1);
}