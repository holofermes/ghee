#include <GLES2/gl2.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <jpeglib.h>
#include "engine.h"

GLuint ghee_load_jpeg_texture(const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "ghee: cannot open JPEG %s\n", filename);
        return 0;
    }

    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, file);
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);

    GLenum format;
    if (cinfo.output_components == 1)      format = GL_LUMINANCE;
    else if (cinfo.output_components == 3) format = GL_RGB;
    else if (cinfo.output_components == 4) format = GL_RGBA;
    else {
        fprintf(stderr, "ghee: unsupported JPEG components: %d\n", cinfo.output_components);
        jpeg_finish_decompress(&cinfo);
        jpeg_destroy_decompress(&cinfo);
        fclose(file);
        return 0;
    }

    unsigned char *data = malloc((size_t)cinfo.output_width * cinfo.output_height * cinfo.output_components);
    if (!data) {
        fprintf(stderr, "ghee: out of memory for JPEG data\n");
        jpeg_finish_decompress(&cinfo);
        jpeg_destroy_decompress(&cinfo);
        fclose(file);
        return 0;
    }

    while (cinfo.output_scanline < cinfo.output_height) {
        unsigned char *row = data + (size_t)cinfo.output_scanline * cinfo.output_width * cinfo.output_components;
        jpeg_read_scanlines(&cinfo, &row, 1);
    }

    const int w = cinfo.output_width, h = cinfo.output_height;
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(file);

    GLuint textureID;
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexImage2D(GL_TEXTURE_2D, 0, format, w, h, 0, format, GL_UNSIGNED_BYTE, data);
    GLenum error = glGetError();
    if (error != GL_NO_ERROR)
        fprintf(stderr, "ghee: GL error uploading texture: 0x%x\n", error);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    free(data);
    glBindTexture(GL_TEXTURE_2D, 0);
    return textureID;
}
