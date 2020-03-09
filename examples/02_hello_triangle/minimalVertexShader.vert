#version 330

layout(location=0) in vec4 inPosition;

void main()
{
    gl_Position = vec4(inPosition.xyz,1);
}