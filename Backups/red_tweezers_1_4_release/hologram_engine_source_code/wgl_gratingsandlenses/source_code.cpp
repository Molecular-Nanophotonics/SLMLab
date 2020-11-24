/*
 *  source_code.cpp
 *  this contains the GLSL source code for the fragment shader that we use to render the hologram
 *
 */

// This shader performs the gratings and lenses algorithm
static const char *GLSLSource = {
 "const vec4 white = vec4(1,1,1,1);"
 "const float pi = 3.1415;"
 "uniform sampler2D pattern;"
 "void main(void)"
 "{"
 "   vec2 xy=gl_TexCoord[0].xy;"//current xy position
 "   if(xy.y > 0.25 && xy.y < 0.75) gl_FragColor = texture(pattern, xy * vec2(1.0, 2.0) - vec2(0.0, 0.5));"
 "   else gl_FragColor = vec4(1,1,1,1);"
 "}"
};

static const GLubyte splashgraphics[524288] = {
};