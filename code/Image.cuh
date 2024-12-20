//save a bmp image, do Gaussian blurring or down sampling
#pragma once
#define _CRT_SECURE_NO_WARNINGS

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cassert>

#include "vecmath.cuh"

// some helper functions for save & load

static unsigned char ReadByte(FILE* file)
{
    unsigned char b;
    auto success = fread((void*)(&b), sizeof(unsigned char), 1, file);
    assert(success == 1);
    return b;
}

static void WriteByte(FILE* file, unsigned char b)
{
    auto success = fwrite((void*)(&b), sizeof(unsigned char), 1, file);
    assert(success == 1);
}

static unsigned char ClampColorComponent(float c)
{
    int tmp = int(c * 255);

    if (tmp < 0)
    {
        tmp = 0;
    }

    if (tmp > 255)
    {
        tmp = 255;
    }

    return (unsigned char)tmp;
}

/****************************************************************************
    bmp.c - read and write bmp images.
    Distributed with Xplanet.
    Copyright (C) 2002 Hari Nair <hari@alumni.caltech.edu>

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
****************************************************************************/

struct BMPHeader
{
    char bfType[3];       /* "BM" */
    int bfSize;           /* Size of file in bytes */
    int bfReserved;       /* set to 0 */
    int bfOffBits;        /* Byte offset to actual bitmap data (= 54) */
    int biSize;           /* Size of BITMAPINFOHEADER, in bytes (= 40) */
    int biWidth;          /* Width of image, in pixels */
    int biHeight;         /* Height of images, in pixels */
    short biPlanes;       /* Number of planes in target device (set to 1) */
    short biBitCount;     /* Bits per pixel (24 in this case) */
    int biCompression;    /* Type of compression (0 if no compression) */
    int biSizeImage;      /* Image size, in bytes (0 if no compression) */
    int biXPelsPerMeter;  /* Resolution in pixels/meter of display device */
    int biYPelsPerMeter;  /* Resolution in pixels/meter of display device */
    int biClrUsed;        /* Number of colors in the color table (if 0, use
                             maximum allowed by biBitCount) */
    int biClrImportant;   /* Number of important colors.  If 0, all colors
                             are important */
};

// Simple image class
class Image
{
public:
    Image(int w, int h)
    {
        width = w;
        height = h;
        data = new Vector3f[width * height];
    }

    ~Image()
    {
        delete[] data;
    }

    int Width() const
    {
        return width;
    }

    int Height() const
    {
        return height;
    }

    const Vector3f& GetPixel(int x, int y) const
    {
        assert(x >= 0 && x < width);
        assert(y >= 0 && y < height);
        return data[y * width + x];
    }

    Vector3f* GetData()
    {
        return data;
    }

    void SetAllPixels(const Vector3f& color)
    {
        for (int i = 0; i < width * height; ++i)
        {
            data[i] = color;
        }
    }

    void SetPixel(int x, int y, const Vector3f& color)
    {
        assert(x >= 0 && x < width);
        assert(y >= 0 && y < height);
        data[y * width + x] = color;
    }

    //anti-aliasing
    void GaussianBlur()
    {
        float kernel[5] = { 0.1201f, 0.2339f, 0.2931f, 0.2339f, 0.1201f };

        //horizontal pass
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                Vector3f sum(0, 0, 0);
                for (int i = -2; i <= 2; ++i)
                {
                    int xIndex = x + i;
                    if (xIndex < 0)
                    {
                        sum = sum + data[y * width] * kernel[i + 2];
                    }
                    else if (xIndex >= width)
                    {
                        sum = sum + data[y * width + width - 1] * kernel[i + 2];
                    }
                    else
                    {
                        sum = sum + data[y * width + xIndex] * kernel[i + 2];
                    }
                }
                data[y * width + x] = sum;
            }
        }

        // Vertical pass
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                Vector3f sum(0, 0, 0);
                for (int i = -2; i <= 2; ++i)
                {
                    int yIndex = y + i;
                    if (yIndex < 0)
                    {
                        sum = sum + data[x] * kernel[i + 2];
                    }
                    else if (yIndex >= height)
                    {
                        sum = sum + data[(height - 1) * width + x] * kernel[i + 2];
                    }
                    else
                    {
                        sum = sum + data[yIndex * width + x] * kernel[i + 2];
                    }
                }
                data[y * width + x] = sum;
            }
        }
    }
    void DownSampling(const Image& image)
    {
        printf("Start down sampling (%d, %d) => (%d, %d)\n", image.width, image.height, width, height);
        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                Vector3f color(0, 0, 0);
                //manual pragma unroll
                color = color + image.GetPixel(x * 3, y * 3);
                color = color + image.GetPixel(x * 3 + 1, y * 3);
                color = color + image.GetPixel(x * 3 + 2, y * 3);

                color = color + image.GetPixel(x * 3, y * 3 + 1);
                color = color + image.GetPixel(x * 3 + 1, y * 3 + 1);
                color = color + image.GetPixel(x * 3 + 2, y * 3 + 1);

                color = color + image.GetPixel(x * 3, y * 3 + 2);
                color = color + image.GetPixel(x * 3 + 1, y * 3 + 2);
                color = color + image.GetPixel(x * 3 + 2, y * 3 + 2);

                color = color / 9.0;

                data[y * width + x] = color;
            }
        }
    }

    // Save and Load PPM image files using magic number 'P6' and having one comment line
    static Image* LoadPPM(const char* filename)
    {
        assert(filename != NULL);
        // must end in .ppm
        const char* ext = &filename[strlen(filename) - 4];
        assert(!strcmp(ext, ".ppm"));
        FILE* file = fopen(filename, "rb");
        // misc header information
        int width = 0;
        int height = 0;
        char tmp[100];
        fgets(tmp, 100, file);
        assert(strstr(tmp, "P6"));
        fgets(tmp, 100, file);
        assert(tmp[0] == '#');
        fgets(tmp, 100, file);
        sscanf(tmp, "%d %d", &width, &height);
        fgets(tmp, 100, file);
        assert(strstr(tmp, "255"));
        // the data
        Image* answer = new Image(width, height);
        // flip y so that (0,0) is bottom left corner
        for (int y = height - 1; y >= 0; y--) {
            for (int x = 0; x < width; x++) {
                unsigned char r, g, b;
                r = fgetc(file);
                g = fgetc(file);
                b = fgetc(file);
                Vector3f color(r / 255.0f, g / 255.0f, b / 255.0f);
                answer->SetPixel(x, y, color);
            }
        }
        fclose(file);
        return answer;
    }
    void SavePPM(const char* filename) const
    {
        assert(filename != NULL);
        // must end in .ppm
        const char* ext = &filename[strlen(filename) - 4];
        assert(!strcmp(ext, ".ppm"));
        FILE* file = fopen(filename, "w");
        // misc header information
        assert(file != NULL);
        fprintf(file, "P6\n");
        fprintf(file, "# Creator: Image::SavePPM()\n");
        fprintf(file, "%d %d\n", width, height);
        fprintf(file, "255\n");
        // the data
        // flip y so that (0,0) is bottom left corner
        for (int y = height - 1; y >= 0; y--)
        {
            for (int x = 0; x < width; x++)
            {
                Vector3f v = GetPixel(x, y);
                fputc(ClampColorComponent(v[0]), file);
                fputc(ClampColorComponent(v[1]), file);
                fputc(ClampColorComponent(v[2]), file);
            }
        }
        fclose(file);
    }

    static Image* LoadTGA(const char* filename)
    {
        assert(filename != NULL);
        // must end in .tga
        const char* ext = &filename[strlen(filename) - 4];
        assert(!strcmp(ext, ".tga"));
        FILE* file = fopen(filename, "rb");
        // misc header information
        int width = 0;
        int height = 0;
        for (int i = 0; i < 18; i++) {
            unsigned char tmp;
            tmp = ReadByte(file);
            if (i == 2) assert(tmp == 2);
            else if (i == 12) width += tmp;
            else if (i == 13) width += 256 * tmp;
            else if (i == 14) height += tmp;
            else if (i == 15) height += 256 * tmp;
            else if (i == 16) assert(tmp == 24);
            else if (i == 17) assert(tmp == 32);
            else assert(tmp == 0);
        }
        // the data
        Image* answer = new Image(width, height);
        // flip y so that (0,0) is bottom left corner
        for (int y = height - 1; y >= 0; y--)
        {
            for (int x = 0; x < width; x++)
            {
                unsigned char r, g, b;
                // note reversed order: b, g, r
                b = ReadByte(file);
                g = ReadByte(file);
                r = ReadByte(file);
                Vector3f color(r / 255.0f, g / 255.0f, b / 255.0f);
                answer->SetPixel(x, y, color);
            }
        }
        fclose(file);
        return answer;
    }
    void SaveTGA(const char* filename) const
    {
        assert(filename != NULL);
        // must end in .tga
        const char* ext = &filename[strlen(filename) - 4];
        assert(!strcmp(ext, ".tga"));
        FILE* file = fopen(filename, "wb");
        // misc header information
        for (int i = 0; i < 18; i++)
        {
            if (i == 2) WriteByte(file, 2);
            else if (i == 12) WriteByte(file, width % 256);
            else if (i == 13) WriteByte(file, width / 256);
            else if (i == 14) WriteByte(file, height % 256);
            else if (i == 15) WriteByte(file, height / 256);
            else if (i == 16) WriteByte(file, 24);
            else if (i == 17) WriteByte(file, 32);
            else WriteByte(file, 0);
        }
        // the data
        // flip y so that (0,0) is bottom left corner
        for (int y = height - 1; y >= 0; y--)
        {
            for (int x = 0; x < width; x++)
            {
                Vector3f v = GetPixel(x, y);
                // note reversed order: b, g, r
                WriteByte(file, ClampColorComponent(v[2]));
                WriteByte(file, ClampColorComponent(v[1]));
                WriteByte(file, ClampColorComponent(v[0]));
            }
        }
        fclose(file);
    }

    int SaveBMP(const char* filename)
    {
        int i, j, ipos;
        int bytesPerLine;
        unsigned char* line;
        Vector3f* rgb = data;
        FILE* file;
        struct BMPHeader bmph;

        /* The length of each line must be a multiple of 4 bytes */

        bytesPerLine = (3 * (width + 1) / 4) * 4;

        strcpy(bmph.bfType, "BM");
        bmph.bfOffBits = 54;
        bmph.bfSize = bmph.bfOffBits + bytesPerLine * height;
        bmph.bfReserved = 0;
        bmph.biSize = 40;
        bmph.biWidth = width;
        bmph.biHeight = height;
        bmph.biPlanes = 1;
        bmph.biBitCount = 24;
        bmph.biCompression = 0;
        bmph.biSizeImage = bytesPerLine * height;
        bmph.biXPelsPerMeter = 0;
        bmph.biYPelsPerMeter = 0;
        bmph.biClrUsed = 0;
        bmph.biClrImportant = 0;

        file = fopen(filename, "wb");
        if (file == NULL) return(0);

        fwrite(&bmph.bfType, 2, 1, file);
        fwrite(&bmph.bfSize, 4, 1, file);
        fwrite(&bmph.bfReserved, 4, 1, file);
        fwrite(&bmph.bfOffBits, 4, 1, file);
        fwrite(&bmph.biSize, 4, 1, file);
        fwrite(&bmph.biWidth, 4, 1, file);
        fwrite(&bmph.biHeight, 4, 1, file);
        fwrite(&bmph.biPlanes, 2, 1, file);
        fwrite(&bmph.biBitCount, 2, 1, file);
        fwrite(&bmph.biCompression, 4, 1, file);
        fwrite(&bmph.biSizeImage, 4, 1, file);
        fwrite(&bmph.biXPelsPerMeter, 4, 1, file);
        fwrite(&bmph.biYPelsPerMeter, 4, 1, file);
        fwrite(&bmph.biClrUsed, 4, 1, file);
        fwrite(&bmph.biClrImportant, 4, 1, file);

        line = (unsigned char*)malloc(bytesPerLine);
        if (line == NULL)
        {
            fprintf(stderr, "Can't allocate memory for BMP file.\n");
            return(0);
        }

        for (i = 0; i < height; i++)
        {
            for (j = 0; j < width; j++)
            {
                ipos = (width * i + j);
                line[3 * j] = ClampColorComponent(rgb[ipos][2]);
                line[3 * j + 1] = ClampColorComponent(rgb[ipos][1]);
                line[3 * j + 2] = ClampColorComponent(rgb[ipos][0]);
            }
            fwrite(line, bytesPerLine, 1, file);
        }

        free(line);
        fclose(file);

        return(1);
    }
    void SaveImage(const char* filename)
    {
        auto len = strlen(filename);
        if (strcmp(".bmp", filename + len - 4) == 0) {
            SaveBMP(filename);
        }
        else {
            SaveTGA(filename);
        }
    }
    // extension for image comparison
    static Image* compare(Image* img1, Image* img2)
    {
        assert(img1->Width() == img2->Width());
        assert(img1->Height() == img2->Height());

        Image* img3 = new Image(img1->Width(), img1->Height());

        for (int x = 0; x < img1->Width(); x++) {
            for (int y = 0; y < img1->Height(); y++) {
                Vector3f color1 = img1->GetPixel(x, y);
                Vector3f color2 = img2->GetPixel(x, y);
                Vector3f color3 =
                    Vector3f
                    (
                        fabs(color1[0] - color2[0]),
                        fabs(color1[1] - color2[1]),
                        fabs(color1[2] - color2[2])
                    );
                img3->SetPixel(x, y, color3);
            }
        }

        return img3;
    }

private:

    int width;
    int height;
    Vector3f* data;

};
