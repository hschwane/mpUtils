#version 330

in vec2 texCoords;
out vec4 color;

uniform vec4 spriteColor;
uniform sampler2D image;

void main()
{
    color = spriteColor * texture(image, texCoords);
}