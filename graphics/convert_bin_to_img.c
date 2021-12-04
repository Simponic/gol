#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Most BMP Image generation code is from https://stackoverflow.com/a/47785639/15819675

const int BYTES_PER_PIXEL = 3; /// red, green, & blue
const int FILE_HEADER_SIZE = 14;
const int INFO_HEADER_SIZE = 40;

void generateBitmapImage(unsigned char* image, int height, int width, char* imageFileName);
unsigned char* createBitmapFileHeader(int height, int stride);
unsigned char* createBitmapInfoHeader(int height, int width);

void generateBitmapImage (unsigned char* image, int height, int width, char* imageFileName)
{
    int widthInBytes = width * BYTES_PER_PIXEL;

    unsigned char padding[3] = {0, 0, 0};
    int paddingSize = (4 - (widthInBytes) % 4) % 4;

    int stride = (widthInBytes) + paddingSize;

    FILE* imageFile = fopen(imageFileName, "wb");

    unsigned char* fileHeader = createBitmapFileHeader(height, stride);
    fwrite(fileHeader, 1, FILE_HEADER_SIZE, imageFile);

    unsigned char* infoHeader = createBitmapInfoHeader(height, width);
    fwrite(infoHeader, 1, INFO_HEADER_SIZE, imageFile);

    int i;
    for (i = 1; i <= height; i++) {
        fwrite(image + ((height - i)*widthInBytes), BYTES_PER_PIXEL, width, imageFile);
        fwrite(padding, 1, paddingSize, imageFile);
    }

    fclose(imageFile);
}

unsigned char* createBitmapFileHeader (int height, int stride)
{
    int fileSize = FILE_HEADER_SIZE + INFO_HEADER_SIZE + (stride * height);

    static unsigned char fileHeader[] = {
        0,0,     /// signature
        0,0,0,0, /// image file size in bytes
        0,0,0,0, /// reserved
        0,0,0,0, /// start of pixel array
    };

    fileHeader[ 0] = (unsigned char)('B');
    fileHeader[ 1] = (unsigned char)('M');
    fileHeader[ 2] = (unsigned char)(fileSize      );
    fileHeader[ 3] = (unsigned char)(fileSize >>  8);
    fileHeader[ 4] = (unsigned char)(fileSize >> 16);
    fileHeader[ 5] = (unsigned char)(fileSize >> 24);
    fileHeader[10] = (unsigned char)(FILE_HEADER_SIZE + INFO_HEADER_SIZE);

    return fileHeader;
}

unsigned char* createBitmapInfoHeader (int height, int width)
{
    static unsigned char infoHeader[] = {
        0,0,0,0, /// header size
        0,0,0,0, /// image width
        0,0,0,0, /// image height
        0,0,     /// number of color planes
        0,0,     /// bits per pixel
        0,0,0,0, /// compression
        0,0,0,0, /// image size
        0,0,0,0, /// horizontal resolution
        0,0,0,0, /// vertical resolution
        0,0,0,0, /// colors in color table
        0,0,0,0, /// important color count
    };

    infoHeader[ 0] = (unsigned char)(INFO_HEADER_SIZE);
    infoHeader[ 4] = (unsigned char)(width      );
    infoHeader[ 5] = (unsigned char)(width >>  8);
    infoHeader[ 6] = (unsigned char)(width >> 16);
    infoHeader[ 7] = (unsigned char)(width >> 24);
    infoHeader[ 8] = (unsigned char)(height      );
    infoHeader[ 9] = (unsigned char)(height >>  8);
    infoHeader[10] = (unsigned char)(height >> 16);
    infoHeader[11] = (unsigned char)(height >> 24);
    infoHeader[12] = (unsigned char)(1);
    infoHeader[14] = (unsigned char)(BYTES_PER_PIXEL*8);

    return infoHeader;
}

void convertBinToRaw(char* file, int width, int height) {
  unsigned char white[3] = {255,255,255};
  unsigned char black[3] = {0,0,0};

  unsigned char* grid = malloc(sizeof(unsigned char) * width * height);

  FILE* fp = fopen(file, "rb");
  fread(grid, sizeof(unsigned char), width*height, fp); 
  fclose(fp);

  int i;
  char* raw_file_name;
  sprintf(raw_file_name, "%s.bmp", file);
  unsigned char* full_image = malloc(sizeof(unsigned char) * width * height * 3);
  for (i = 0; i < width*height; i++) {
    if (grid[i]) {
      memcpy(&full_image[i*3], &white, sizeof(unsigned char) * 3);
    } else {
      memcpy(&full_image[i*3], &black, sizeof(unsigned char) * 3);
    }
  }

  generateBitmapImage(full_image, height, width, raw_file_name);

  free(grid);
  free(full_image);
}

int main(int argc, char** argv) {
  char* file_name;
  int width, height;
  if (argc == 4) {
    file_name = argv[1];
    width = atoi(argv[2]);
    height = atoi(argv[3]);
  } else {
    printf("Usage: ./convert_bin_to_img <file> <width> <height>\n");
    exit(1);
  }

  convertBinToRaw(file_name, width, height);

  return 0;
}
