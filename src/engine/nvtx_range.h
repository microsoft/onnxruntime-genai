#pragma once

#if defined(ORTGENAI_ENABLE_NVTX)
#include <nvtx3/nvToolsExt.h>

namespace Generators {

struct NvtxRange {
  explicit NvtxRange(const char* name) noexcept { nvtxRangePushA(name); }
  NvtxRange(const NvtxRange&) = delete;
  NvtxRange& operator=(const NvtxRange&) = delete;
  ~NvtxRange() { nvtxRangePop(); }
};

struct NvtxPhases {
  NvtxPhases() = default;
  NvtxPhases(const NvtxPhases&) = delete;
  NvtxPhases& operator=(const NvtxPhases&) = delete;
  ~NvtxPhases() { End(); }

  void Next(const char* name) noexcept {
    End();
    nvtxRangePushA(name);
    open_ = true;
  }

  void End() noexcept {
    if (open_) {
      nvtxRangePop();
      open_ = false;
    }
  }

 private:
  bool open_{false};
};

}  // namespace Generators

#define ORTGENAI_NVTX_CAT_IMPL(first, second) first##second
#define ORTGENAI_NVTX_CAT(first, second) ORTGENAI_NVTX_CAT_IMPL(first, second)
#define ORTGENAI_NVTX_RANGE(name) \
  const ::Generators::NvtxRange ORTGENAI_NVTX_CAT(ortgenai_nvtx_range_, __LINE__) { name }
#define ORTGENAI_NVTX_PHASES(phases) ::Generators::NvtxPhases phases
#define ORTGENAI_NVTX_PHASE_NEXT(phases, name) phases.Next(name)
#else
#define ORTGENAI_NVTX_RANGE(name) ((void)0)
#define ORTGENAI_NVTX_PHASES(phases) ((void)0)
#define ORTGENAI_NVTX_PHASE_NEXT(phases, name) ((void)0)
#endif