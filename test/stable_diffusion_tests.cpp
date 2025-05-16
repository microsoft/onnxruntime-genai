// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <span>
#include <cstring>  // for memcmp
#include <numeric>
#include <random>
#include "span.h"
#define OGA_USE_SPAN 1
#include <ort_genai.h>
#include <gtest/gtest.h>
#include "models/onnxruntime_api.h"

// Our working directory is generators/build so one up puts us in the root directory:
#ifndef MODEL_PATH
#define MODEL_PATH "../../test/test_models/"
#endif

TEST(StableDiffusionTests, ClipTokenizer) {
  auto config = OgaConfig::Create(MODEL_PATH "sd");

  auto model = OgaModel::Create(*config);
  auto tokenizer = OgaTokenizer::Create(*model);

  auto sequences = OgaSequences::Create();
  tokenizer->Encode("Capybara with his mom and dad in a beautiful stream", *sequences);

  std::vector<int32_t> expected_tokens{49406, 1289, 88, 19345, 593, 787, 2543, 537, 2639, 530, 320, 1215, 3322, 49407};

  EXPECT_TRUE(1 == sequences->Count());
  EXPECT_TRUE(sequences->SequenceCount(0) == expected_tokens.size());
  EXPECT_TRUE(0 == std::memcmp(sequences->SequenceData(0), expected_tokens.data(), expected_tokens.size() * sizeof(int32_t)));
}

#pragma pack(push, 1)
struct BITMAPFILEHEADER {
  uint16_t bfType;
  uint32_t bfSize;
  uint16_t bfReserved1;
  uint16_t bfReserved2;
  uint32_t bfOffBits;
};

static_assert(sizeof(BITMAPFILEHEADER) == 14, "BITMAPFILEHEADER size mismatch");

struct BITMAPINFOHEADER {
  uint32_t biSize;
  int32_t biWidth;
  int32_t biHeight;
  uint16_t biPlanes;
  uint16_t biBitCount;
  uint32_t biCompression;
  uint32_t biSizeImage;
  int32_t biXPelsPerMeter;
  int32_t biYPelsPerMeter;
  uint32_t biClrUsed;
  uint32_t biClrImportant;
};

static_assert(sizeof(BITMAPINFOHEADER) == 40, "BITMAPINFOHEADER size mismatch");

#pragma pack(pop)

TEST(StableDiffusionTests, StableDiffusion) {
  // auto config = OgaConfig::Create(MODEL_PATH "sd");

  // auto model = OgaModel::Create(*config);
  // auto tokenizer = OgaTokenizer::Create(*model);

  // auto sequences = OgaSequences::Create();
  // tokenizer->Encode("Capybara with his mom and dad in a beautiful stream", *sequences);

  OgaTensor* images;
  uint8_t* image_data;
  //OgaCheckResult(OgaSelenaTest("Capybara with his mom and dad in a beautiful stream", MODEL_PATH "sd", &image_data));
  OgaCheckResult(OgaSelenaTest("There are two cats playing in couch", MODEL_PATH "sd", &image_data));

  size_t rank = 4;
  //OgaCheckResult(OgaTensorGetShapeRank(images, &rank));
  std::vector<int64_t> shape{1, 512, 512, 3};
  //OgaCheckResult(OgaTensorGetShape(images, shape.data(), rank));

  //OgaCheckResult(OgaTensorGetData(images, &image_data));

  BITMAPFILEHEADER bfh;
  bfh.bfType = 0x4D42;  // 'BM'
  bfh.bfSize = uint32_t(sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + shape[1] * shape[2] * shape[3]);
  bfh.bfReserved1 = 0;
  bfh.bfReserved2 = 0;
  bfh.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);

  BITMAPINFOHEADER bih;
  bih.biSize = sizeof(BITMAPINFOHEADER);
  bih.biWidth = static_cast<int32_t>(shape[2]);
  bih.biHeight = static_cast<int32_t>(shape[1]);
  bih.biPlanes = 1;
  bih.biBitCount = 24;
  bih.biCompression = 0;  // BI_RGB
  bih.biSizeImage = 0;
  bih.biXPelsPerMeter = 2835;
  bih.biYPelsPerMeter = 2835;
  bih.biClrUsed = 0;
  bih.biClrImportant = 0;

  std::vector<uint8_t> image_bitmap_data(bfh.bfSize);
  std::memcpy(image_bitmap_data.data(), &bfh, sizeof(BITMAPFILEHEADER));
  std::memcpy(image_bitmap_data.data() + sizeof(BITMAPFILEHEADER), &bih, sizeof(BITMAPINFOHEADER));

  // Reverse the rows of the image data
  size_t row_size = shape[2] * shape[3];                          // Width * Channels (e.g., 3 for RGB)
  std::vector<uint8_t> reversed_image_data(shape[1] * row_size);  // Height * Row Size
  for (size_t row = 0; row < size_t(shape[1]); ++row) {
    std::memcpy(
        reversed_image_data.data() + row * row_size,
        static_cast<uint8_t*>(image_data) + (shape[1] - 1 - row) * row_size,
        row_size);
  }

  for (size_t row = 0; row < size_t(shape[1]); ++row) {
    for (size_t col = 0; col < size_t(shape[2]); ++col) {
      // Swap the red and blue channels
      uint8_t* pixel = static_cast<uint8_t*>(reversed_image_data.data()) + (row * shape[2] + col) * shape[3];
      std::swap(pixel[0], pixel[2]);
    }
  }

  // Copy the reversed image data into the bitmap data
  std::memcpy(
      image_bitmap_data.data() + sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER),
      reversed_image_data.data(),
      shape[1] * row_size);
  // HANDLE hFile = CreateFileA("test.bmp", GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
  // if (hFile != INVALID_HANDLE_VALUE) {
  //   DWORD dwWritten;
  //   WriteFile(hFile, image_bitmap_data.data(), static_cast<DWORD>(image_bitmap_data.size()), &dwWritten, NULL);
  //   CloseHandle(hFile);
  // } else {
  //   std::cerr << "Failed to create file" << std::endl;
  // }

  FILE* file = fopen("test.bmp", "wb");
  if (file) {
    fwrite(image_bitmap_data.data(), sizeof(uint8_t), image_bitmap_data.size(), file);
    fclose(file);
  } else {
    std::cerr << "Failed to create file" << std::endl;
  }

  //OgaDestroyTensor(images);
}
