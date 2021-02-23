#version 330

in vec4 color; // linear interpolated color from the vertex shader

out vec4 fragment_color; // color this fragment will have

void main()
{
    fragment_color = color;
}