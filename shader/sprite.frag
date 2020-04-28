#version 440
#extension GL_ARB_bindless_texture : require

in vec4 color;
flat in uvec2 texAdr;
in vec2 texCoords;
out vec4 out_color;

uniform bool alphaDiscard;

void main()
{
    sampler2D spriteTex = sampler2D( texAdr );
    vec4 resultColor = color * texture( spriteTex, texCoords);
    if(alphaDiscard && resultColor.a < 0.5f)
        discard;
    out_color = resultColor;
}