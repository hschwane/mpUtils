#version 450

struct spriteData
{
    mat4 model;
    vec4 color;
    uvec2 bindlessTexture;
    float tileFactor;
};

layout(std430,binding=0) buffer spriteDataSSBO
{
    spriteData sprites[];
};

uniform mat4 projection;

out vec2 texCoords;
out vec4 color;
flat out uvec2 texAdr;

void main()
{
    // load data fro ubo
    const int uboId = gl_VertexID/6;
    color = sprites[uboId].color;
    texAdr = sprites[uboId].bindlessTexture;

    // generate vertex position
    int idInQuad = gl_VertexID%6;
    idInQuad = (idInQuad==4) ? 2 : idInQuad;
    float x = -1 +  float((idInQuad & 1) << 1);
    float y = -1 +  float( (idInQuad & 2));

    // transform and generate texture coordinates
    gl_Position = projection * sprites[uboId].model * vec4(x,y,0,1);
    texCoords = vec2((x+1)/2, (y+1)/2)*sprites[uboId].tileFactor;
}
