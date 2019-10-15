#version 330

uniform mat4 model;
uniform mat4 projection;

out vec2 texCoords;

void main()
{
    float x = -1 +  float((gl_VertexID & 1) << 1);
    float y = -1 +  float((gl_VertexID & 2));

    gl_Position = projection * model * vec4(x,y,0,1);
    texCoords = vec2((x+1)/2, (y+1)/2);
}
