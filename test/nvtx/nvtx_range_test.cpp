#if defined(ORTGENAI_NVTX_TEST_MOCK)
#define NVTX_NO_IMPL
#endif
#include "../../src/engine/nvtx_range.h"

#include <type_traits>

#if defined(ORTGENAI_NVTX_TEST_MOCK)
namespace {
int pushes = 0;
int pops = 0;
int depth = 0;
int max_depth = 0;
bool invalid_pop = false;
}  // namespace

extern "C" int NVTX_API nvtxRangePushA(const char* name) {
  (void)name;
  ++pushes;
  ++depth;
  if (depth > max_depth) max_depth = depth;
  return depth - 1;
}

extern "C" int NVTX_API nvtxRangePop() {
  ++pops;
  if (depth == 0) invalid_pop = true;
  return --depth;
}
#endif

int main() {
  int labels = 0;
  auto label = [&]() {
    ++labels;
    return "test";
  };
  try {
    ORTGENAI_NVTX_RANGE(label());
    ORTGENAI_NVTX_RANGE(label());
    ORTGENAI_NVTX_PHASES(phases);
    ORTGENAI_NVTX_PHASE_NEXT(phases, label());
    ORTGENAI_NVTX_PHASE_NEXT(phases, label());
    throw 1;
  } catch (int) {
  }
#if defined(ORTGENAI_ENABLE_NVTX)
  static_assert(!std::is_copy_constructible_v<Generators::NvtxRange>);
  static_assert(!std::is_move_constructible_v<Generators::NvtxRange>);
  static_assert(!std::is_copy_constructible_v<Generators::NvtxPhases>);
  static_assert(!std::is_move_constructible_v<Generators::NvtxPhases>);
  Generators::NvtxPhases empty;
  empty.End();
  empty.End();
  auto early_return = []() {
    ORTGENAI_NVTX_RANGE("outer");
    ORTGENAI_NVTX_PHASES(phases);
    ORTGENAI_NVTX_PHASE_NEXT(phases, "first");
    phases.End();
    phases.End();
    ORTGENAI_NVTX_PHASE_NEXT(phases, "second");
    return;
  };
  early_return();
#if defined(ORTGENAI_NVTX_TEST_MOCK)
  if (pushes != 7 || pops != 7 || depth != 0 || max_depth != 3 || invalid_pop) return 1;
#endif
  return labels == 4 ? 0 : 1;
#else
  (void)label;
  return labels == 0 ? 0 : 1;
#endif
}