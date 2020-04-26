#version 440
#extension GL_ARB_bindless_texture : require

in vec4 color;
flat in uvec2 texAdr;
in vec2 texCoords;
out vec4 out_color;

void main()
{
    sampler2D spriteTex = sampler2D( texAdr );
    out_color = color * texture( spriteTex, texCoords);
}