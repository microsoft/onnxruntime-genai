#include "gtest/gtest.h"
#include "httplib.h"
#include "ort_genai.h"

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);

  auto status = RUN_ALL_TESTS();

  OgaShutdown();
  return status;
}
