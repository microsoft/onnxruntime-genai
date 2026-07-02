# Plan: Plugin-EP shared allocators, clean `OgaShutdown`, and re-initialization

Status: Draft for review. Two related changes: (a) let genai use plugin-EP
shared allocators instead of the dummy-`OrtSession` allocator bootstrap, and
(b) make `OgaShutdown` a clean teardown so genai can be re-initialized
in-process without a restart (the Foundry Local `Manager` recreate scenario).

Scope is **all changes required to make this work for every EP** — WebGPU,
CUDA / NvTensorRtRtx, DML, QNN, OpenVINO, RyzenAI, VitisAI. The work is
delivered in **stages** (§14): an EP-agnostic re-init foundation first, then
the shared-allocator plumbing, then per-EP migrations. WebGPU and CUDA are the
first EPs migrated; the rest follow the same pattern in later stages, each
independently shippable and testable.

Terminology used throughout:

- **genai CUDA add-on library** — `onnxruntime-genai-cuda.{dll,so}`, built
  from this repo's `src/cuda/` sources
  ([CMakeLists.txt#L195](../CMakeLists.txt#L195)) and loaded by
  [`GetCudaInterface`](../src/generators.cpp#L200). It provides the CUDA
  `DeviceInterface` implementation (device memory, host-copy routines,
  kernels, the CUDA stream). It is a genai-level construct.
- **ORT CUDA EP** — the ONNX Runtime `CUDAExecutionProvider`, registered on
  the `OrtEnv` (as a plugin EP or via the provider bridge). It is the source
  of the shared allocator.

These are independent: the add-on library is *not* the EP, and their
lifetimes are decoupled.

## 1. Background

Today, every non-CPU `DeviceInterface` is bootstrapped via
[`EnsureDeviceOrtInit`](../src/models/model.cpp#L400):

1. Construct a tiny throw-away `OrtSession` with the EP appended.
2. Call `Ort::Allocator::Create(session, mem_info)` to materialize an allocator from that session.
3. Stash `{session, allocator}` in
   [`OrtGlobals::device_allocators_[type]`](../src/generators.h#L157) so the
   allocator outlives any `Model`.
4. Call `device.InitOrt(*Ort::api, *allocator)` and let the device cache the
   raw `Ort::Allocator*` in a file-static / namespace-static.

This was the only option before the plugin-EP API. With plugin EPs the EP
itself adds its allocators to the `OrtEnv` at register time, so the dummy
session is no longer needed. Cleanup is anchored on `OgaShutdown`, not on
unregistration — a host can unregister on its own `OrtEnv` reference, which
genai cannot observe, so genai never treats unregistration as a cleanup signal
(§8).

## 2. Goals

1. **Re-initialization (primary driver).** After `OgaShutdown`, genai returns
   to a just-loaded state; a subsequent genai call re-initializes cleanly
   with a fresh `OrtEnv`. This unblocks hosts that recreate their wrapper
   (Foundry Local `Manager`) and expect fresh env configuration (logging,
   etc.) rather than the stale singleton settings.
2. **Shared allocators.** When a device's plugin EP is registered, genai uses
   the env's shared allocator (fetched on demand) instead of a dummy session.
3. **Clean teardown.** `OgaShutdown` is a thorough teardown — it destroys the
   env and unloads the genai add-on libraries so no cached env-derived state
   survives between cycles.
4. **Keep the legacy path** for non-plugin usage (the provider-bridge CUDA
   path). genai chooses plugin vs legacy *per model* from a live
   `GetEpDevices` lookup (§6) — plugin mode when the EP is registered as a
   plugin EP on the env, legacy otherwise — not from a build-time switch.
5. No behavioral change for DML / QNN / OpenVINO / RyzenAI / NvTensorRtRtx
   beyond what is stated here.

## 3. Teardown and re-initialization model

`OrtEnv` is a process-wide, refcounted singleton. genai's
[`OrtGlobals`](../src/generators.cpp#L65) holds one reference. A host such as
Foundry Local may create the env first (to control logging) and hold its own
reference. The supported teardown order (FL's flow) is:

1. Host destroys all models / generators / tensors (everything holding a
   `DeviceBuffer`).
2. Host calls `OgaShutdown`. genai resets `OrtGlobals`, dropping its internal
   env reference and tearing down its add-on libraries.
3. Host *optionally* unregisters — on its own env reference — the EPs it
   originally registered on that same reference (after `OgaShutdown`, so all
   genai usage is finished). Not required — step 4 unloads the EP libraries
   anyway — so most hosts skip it (§8).
4. Host releases its own env reference. The last drop destroys the env; ORT's
   `Environment` destructor then clears shared allocators and unregisters /
   unloads any still-registered EP libraries automatically.
5. Host may now recreate its wrapper; the next genai call builds a fresh
   `OrtEnv` and re-initializes.

**Confirmed** against ORT: creating a new env after the previous one is
destroyed works, and EP libraries are gracefully unloaded on env teardown — so
ORT plugin EP libraries need no genai-side lifecycle handling. (This is the one
place we state that.) The only genai-side decision is what to do with the genai
*add-on* library.

**Teardown approach: unload the add-on libraries.** `OgaShutdown` resets
`OrtGlobals` (destroying the env) *and* unloads the genai add-on libraries
(starting with CUDA). Unloading runs the add-on DLL's static destructors, so
the interface object, the CUDA stream, and the file-static allocator pointer
all cease to exist between cycles — genai returns to a just-loaded state with
no cached env-derived state and no reset hook. This requires moving the
`LibraryHandle` and interface pointer out of the `GetCudaInterface`
function-statics ([generators.cpp#L200](../src/generators.cpp#L200)) into
`OrtGlobals`.

> *Future optimization:* keep the add-on library resident and recreate only
> the interface each cycle (re-calling `GetInterface`, which frees the old
> `g_cuda_device` + stream and makes fresh ones). Avoids DLL unload/reload but
> is more subtle; unloading is the simpler starting point, to revisit once
> re-init is proven.

**Ownership refactor.** To make teardown a single site, move all state the env
cycle owns into `OrtGlobals` (concrete task list in §14 stage 0):

- **`GetOrtGlobals()` lazily re-creates when null.** Today it initializes a
  function-static once and `Shutdown()` resets it, so re-init is impossible;
  change it to rebuild `OrtGlobals` on demand (guarding against resurrection in
  the process-exit `EnsureShutdown` destructor).
- **`OrtGlobals` owns the `DeviceInterface` instances**, created on first use
  of each EP via a get-or-create accessor under a lock — *not* `std::call_once`
  (process-once, which is exactly why `GetRyzenAIInterface` can't re-init
  today). This replaces the scattered function/namespace statics and lets
  `~OrtGlobals` drop every interface, so the hard-coded
  `RyzenAIInterface::Shutdown()` in `Shutdown()` goes away.
- **DLL-provided interfaces (CUDA)** are owned by the add-on library
  (`g_cuda_device` inside `onnxruntime-genai-cuda`) and handed to genai as a
  raw pointer, so `OrtGlobals` holds the add-on `LibraryHandle` + a non-owning
  pointer; unloading the handle destroys the interface. In-process interfaces
  (CPU / WebGPU / QNN / OpenVINO / RyzenAI) are owned directly.
- **`~OrtGlobals` sequences env destruction vs interface teardown**
  deliberately (e.g. RyzenAI's EP shutdown currently runs after the env is
  gone).
- **Bootstrapping:** `OrtGlobals`'s constructor calls
  `GetDeviceInterface(CPU)->InitOrt(...)` via a member on `this`, not the free
  `GetDeviceInterface` → `GetOrtGlobals()` path, to avoid re-entering
  `GetOrtGlobals()` mid-construction.

CPU needs no teardown handling (process-global allocator, §9). RyzenAI needs
more than the ownership move to become recreatable — see §13.

## 4. ORT API surface we actually need

Per current ORT design the shared allocators are added to the env at
`RegisterExecutionProviderLibrary` time and removed at
`UnregisterExecutionProviderLibrary` time. We do **not** need
`CreateSharedAllocator` / `ReleaseSharedAllocator` (they exist on `Ort::Env`
upstream, but env-managed registration covers our use). We add two thin
free-function wrappers in the existing genai `Ort::` style (raw C handles,
matching `Ort::GetEpDevices`), plus reuse one existing wrapper:

- `Ort::GetEpDevices(env)` — the `OrtEpDevice` list for the env. **Existing**,
  in [`onnxruntime_inline.h`](../src/models/onnxruntime_inline.h#L138).
- `Ort::GetMemoryInfo(const OrtEpDevice*, OrtDeviceMemoryType)` — **new**,
  wraps `OrtApi::EpDevice_MemoryInfo`. Called for `OrtDeviceMemoryType_DEFAULT`
  (device-local) and, when the EP advertises pinned host memory, for
  `OrtDeviceMemoryType_HOST_ACCESSIBLE`. Returns a non-owning
  `const OrtMemoryInfo*`.
- `Ort::GetSharedAllocator(OrtEnv*, const OrtMemoryInfo*)` — **new**, wraps
  `OrtApi::GetSharedAllocator`. Returns a non-owning `Ort::Allocator*` owned by
  the env.

## 5. Lifetime contract (documented, not enforced in v1)

> No `Model`, `Generator`, `Tokenizer`, `OgaTensor`, `Engine`, or any object
> that holds a `DeviceBuffer` may outlive `OgaShutdown`.
>
> The caller must destroy every such object before calling `OgaShutdown`.
> Calling `OgaShutdown` with live device buffers is undefined behavior
> (typically a crash in the buffer's destructor when it frees through a
> now-invalid allocator, or a dangling `DeviceInterface*`).

Because the guaranteed release point is `OgaShutdown` (explicit per-EP
unregister is also possible — §8), the contract is stated in terms of
shutdown. Debug builds may assert that no `DeviceBuffer`s are outstanding at
shutdown; no production refcounting of the allocator is introduced.

## 6. Detecting plugin vs legacy mode via `GetEpDevices`

A CUDA or WebGPU `DeviceInterface` can be reached two ways at runtime:

- as a **plugin EP** registered on the env (device-local allocator lives on
  the env), or
- via the **legacy path** — the provider-bridge path, or any EP brought
  in without plugin registration — where the allocator comes from the
  dummy-session bootstrap.

The authoritative signal for which mode applies is simply **whether the
interface's EP name appears in `Ort::GetEpDevices(env)`**. Present → plugin
mode (use the env's shared allocator); absent → legacy mode
(`EnsureDeviceOrtInit`). Each interface knows its own EP name(s), so it looks
*itself* up. An EP name can appear on **several** `OrtEpDevice` entries (e.g.
multi-GPU), so the interface gathers them all rather than stopping at the
first:

```cpp
std::vector<const OrtEpDevice*> FindMyEpDevices(OrtEnv& env) {
  std::vector<const OrtEpDevice*> devices;
  for (const auto* d : Ort::GetEpDevices(env))
    if (std::strcmp(Ort::EpName(d), kMyEpName) == 0)  // e.g. "CUDAExecutionProvider"
      devices.push_back(d);
  return devices;  // empty -> legacy mode
}
```

From the matched devices the interface collects the distinct `OrtMemoryInfo`s
they advertise (`DEFAULT` device-local and, if present, `HOST_ACCESSIBLE`) and
fetches one shared allocator per distinct mem-info via `GetSharedAllocator`.
Two devices may advertise the **same** `OrtMemoryInfo` (a shared allocator),
so the collection **de-duplicates by mem-info** — fetching by an identical
mem-info just returns the same env allocator, so a set keyed on mem-info
avoids redundant entries.

Consequences:

- This replaces a `HasSharedAllocator()` flag: mode is derived from live env
  state, not from a stored bit, so it is automatically correct across
  register / shutdown / re-init.
- It removes the need for an `OnEpRegistered` "push" and a registration-time
  bind walk. The interface **pulls** what it needs (device → mem-info →
  shared allocator) lazily, the first time it allocates.
- No central EP-name → `DeviceType` registry is needed, because each
  interface matches only its own name(s).

The lookup result (the matched devices and their distinct mem-infos / shared
allocators, and hence the mode) may be memoized for the current env cycle.
Since the interface is rebuilt each cycle (§3), the memo never outlives its
env and needs no explicit clearing.

## 7. `DeviceInterface` changes

Source: [`smartptrs.h`](../src/smartptrs.h#L101).

- **No cached allocator.** For plugin mode, fetch the allocator on demand via
  `GetSharedAllocator(env, mem_info)`. The env owns it and it is valid for
  exactly the window the EP is registered — which, per §5, covers every live
  `DeviceBuffer`. This removes the file-static `ort_allocator_` and the
  `assert(!ort_allocator_)` re-init blocker.
- **Mode + mem-infos via `GetEpDevices`** (§6): enumerate all matching
  devices, collect their distinct mem-infos, memoized per env cycle.
- **Keep `InitOrt(const OrtApi&, Ort::Allocator&)`** for the legacy path
  (provider-bridge CUDA, DML, anything still going through
  `EnsureDeviceOrtInit`). It must no longer assert on re-entry.
- **`GetHostAccessibleAllocator()`** (default `nullptr`): for EPs that expose
  a pinned / mappable host allocator (CUDA host-pinned staging). Fetched on
  demand from the host-accessible mem-info when present (§11).
- **No `OnEpUnregistering` and no `HasSharedAllocator` flag** — both are
  obsoleted by on-demand fetch + `GetEpDevices` detection + shutdown-only
  release.

`DeviceBuffer` keeps its current shape: it captures the allocator pointer at
construction (now sourced from `GetSharedAllocator` in plugin mode) and frees
in its dtor. Both happen while the EP is registered, so the pointer is valid
throughout; no refcounting is introduced.

## 8. Registration / unregistration plumbing

In [`ort_genai_c.cpp`](../src/ort_genai_c.cpp#L1116).

- **`OgaRegisterExecutionProviderLibrary(name, path)`** forwards to
  `Ort::RegisterExecutionProviderLibrary` as today — no bind walk, no
  bookkeeping. ORT adds the EP's shared allocators to the env; interfaces
  discover themselves lazily via `GetEpDevices` (§6).
- **`OgaUnregisterExecutionProviderLibrary(name)`** forwards to
  `Ort::UnregisterExecutionProviderLibrary` — an explicit unregister should
  actually unregister (least surprise). It uses genai's *internal* env
  reference, so it must be called **before** `OgaShutdown` and only once **all
  usage of that EP is finished** (no live object holds a `DeviceBuffer` from
  it). See the ordering rules below.

**Ordering rules.** There are potentially two `OrtEnv` references: genai holds one
*internally* (dropped at `OgaShutdown`), and a host such as Foundry Local may
hold its *own*. genai never uses unregistration as a cleanup signal — a host
can unregister on its own reference, which genai cannot observe — so cleanup
of genai's env-derived state is anchored on `OgaShutdown`. Rules:

- **Registration may happen at any time**, including *after* genai is already
  in use: when a model that needs an EP is created, genai discovers that EP's
  allocator lazily via `GetEpDevices` (§6), so there is no "register before
  first use" requirement. (Foundry Local happens to register everything up
  front, before any genai usage.)
- **Unregister an EP only when it is no longer in use** — after every object
  holding a `DeviceBuffer` from it has been destroyed. Unregistering with live
  buffers drops the env's shared allocator out from under them (dangling
  free). Timing relative to `OgaShutdown` depends on which env reference is
  used:
  - **Direct-ORT (host's own reference):** the reference outlives
    `OgaShutdown`, so unregister **after** `OgaShutdown` — i.e. after all genai
    usage is finished. This is Foundry Local's flow.
  - **genai path (`OgaUnregisterExecutionProviderLibrary`):** uses genai's
    internal reference, dropped at `OgaShutdown`, so it must run **before**
    `OgaShutdown`. Optional — skipping it lets env destruction unregister the
    remaining EP libraries (§3).

This is the companion to the §5 lifetime contract.

**Pick one method per EP** so you don't double-register at the ORT level;
genai keeps no registration bookkeeping and pairs nothing, because mode
detection is a live `GetEpDevices` lookup (§6) that works regardless of *how*
an EP was registered:

- **Direct-ORT:** host registers / unregisters on its own env reference.
  Foundry Local uses this — register before any genai usage, unregister after
  `OgaShutdown`.
- **genai path:** host calls `OgaRegisterExecutionProviderLibrary` /
  `OgaUnregisterExecutionProviderLibrary` (the latter before `OgaShutdown`,
  optional).

Built-in / statically linked EPs (CPU) never appear as plugin devices and fall
through to their normal handling.

## 9. CPU allocator

CPU is not a plugin EP and needs no special teardown. `OrtGlobals` registers
an arena over the CPU mem-info on the env via `CreateAndRegisterAllocator`,
but the CPU `DeviceInterface` allocates through
[`Ort::Allocator::GetWithDefaultOptions()`](../src/models/onnxruntime_api.h#L428),
which is the process-global default allocator owned by ONNX Runtime — **not**
env-scoped. So:

- The pointer the CPU interface caches does not dangle when the env is
  destroyed; nothing genai holds points into the arena.
- On re-init the new `OrtGlobals` ctor re-runs `CreateAndRegisterAllocator`
  on the new env and re-calls `InitOrt` with the same process-global default
  allocator.

The only change CPU needs for re-init is to make `InitOrt` idempotent — drop
the `assert(!ort_allocator_)` at
[`cpu/interface.cpp#L52`](../src/cpu/interface.cpp#L52), since it is called
again with the same pointer. No clearing, no rebind.

## 10. `EnsureDeviceOrtInit` and `OrtGlobals::device_allocators_`

Keep both, narrowed to the legacy path:

- Plugin mode (§6) bypasses `EnsureDeviceOrtInit` and fetches the shared
  allocator on demand.
- The provider-bridge CUDA path
  ([`cuda/session_options.cpp`](../src/cuda/session_options.cpp#L66))
  continues to call `EnsureDeviceOrtInit` and populate
  `OrtGlobals::device_allocators_[CUDA]`. Retiring this requires removing the
  provider-bridge path, a separate decision.

`Model` construction selects the path by the §6 check: if
`p_device_->FindMyEpDevices(env)` is non-empty it is plugin mode (shared
allocator fetched on demand, nothing to bootstrap); otherwise it calls
`EnsureDeviceOrtInit(*p_device_, *config_)` (legacy path), which keeps its
"early-return if already populated" behavior.
Once CUDA's provider-bridge path is retired, both `EnsureDeviceOrtInit` and
`OrtGlobals::device_allocators_` can be deleted.

## 11. Host-accessible allocator

The rule is EP-agnostic. On demand, genai looks for a `HOST_ACCESSIBLE`
mem-info among the EP's matched devices (§6;
`Ort::GetMemoryInfo(dev, OrtDeviceMemoryType_HOST_ACCESSIBLE)`, §4):

- **Found** → `GetHostAccessibleAllocator()` returns the env's shared
  allocator for it, and genai uses that for host-side staging (`AllocateCpu`).
- **Not found** → it returns `nullptr` and callers take the EP-agnostic
  fallback: a plain host staging buffer with device↔CPU copies via
  `OrtEnv::CopyTensors`.

No per-EP special-casing. As examples: the CUDA plugin EP advertises a
`HOST_ACCESSIBLE` allocator (`CUDAPinnedAllocator`, used for
[`GpuMemory::AllocateCpu`](../src/cuda/interface.cpp#L46)); WebGPU advertises
none and so takes the `CopyTensors` fallback. The calling code is identical
either way.

## 12. Per-EP impact (in scope)

### 12.1 WebGPU

Files: [`webgpu/interface.cpp`](../src/webgpu/interface.cpp),
[`webgpu/session_options.cpp`](../src/webgpu/session_options.cpp).

- Allocator fetched on demand; mode + device-local mem-info discovered via
  `GetEpDevices` (§6).
- Remove the cached `ort_allocator_` / `assert(!ort_allocator_)`; `InitOrt`
  becomes a no-op / unreachable for WebGPU (plugin path is the only WebGPU
  path).
- Delete the WebGPU branches in `EnsureDeviceOrtInit` once no callers remain.

### 12.2 CUDA / NvTensorRtRtx

Files: [`cuda/interface.cpp`](../src/cuda/interface.cpp),
[`cuda/session_options.cpp`](../src/cuda/session_options.cpp),
[`nvtensorrtrtx/session_options.cpp`](../src/nvtensorrtrtx/session_options.cpp).

- Remove the file-static `ort_allocator_`; fetch on demand in plugin mode.
  `device_label` stays (it identifies the buffer type). `GpuMemory` captures
  the allocator at construction from the interface.
- Discover the host-accessible mem-info (if any) via `GetEpDevices` and use
  its shared allocator for `AllocateCpu` / free (§11); in plugin mode CUDA
  always advertises one. The direct `::cudaHostAlloc` / `cudaFreeHost` calls
  remain only as the legacy-mode (provider-bridge) fallback.
- The CUDA stream and kernels are genai add-on-library state, **not**
  EP-scoped: they are created with the interface and destroyed only when the
  add-on library is torn down at `OgaShutdown` (§3). They are not touched by
  EP registration state.
- Keep `InitOrt` for the provider-bridge path; the two paths are mutually
  exclusive at runtime and selected by the §6 check.
- `cuda/session_options.cpp::AppendExecutionProvider` keeps its "try V2
  plugin, fall back to provider-bridge" logic.

## 13. Per-EP notes and non-goals

All EPs are in scope (staged in §14). Per-EP specifics:

- **WebGPU, CUDA / NvTensorRtRtx.** First migrations; detailed in §12.
- **DML.** Currently on the trivial-session + `g_dml_device` +
  `CloseDmlInterface` path. Migrating means moving it onto the
  `GetEpDevices` detection + shared-allocator model like the others; its
  existing `Model::~Model` teardown folds into `~OrtGlobals` (§3).
- **QNN, OpenVINO.** Migrate to the shared-allocator path once each is
  exercised as a plugin EP. OpenVINO is the motivating multi-EP-name case
  (a single library exposing several EP names); the §6 per-interface
  self-lookup handles that without a central registry.
- **RyzenAI.** Today `GetRyzenAIInterface()` self-registers the EP on first
  call (fallback search for `onnxruntime_providers_ryzenai.dll` next to
  `onnxruntime-genai.dll`, `onnxruntime.dll`, the executable, or CWD; see
  [`ryzenai/interface.cpp`](../src/ryzenai/interface.cpp)), and the genai
  binary is built with RyzenAI sources unconditionally (cmake glob in
  [`global_variables.cmake`](../cmake/global_variables.cmake#L78)). Making it
  recreatable is more than the §3 ownership move: its ctor early-returns when
  the EP module is already resident (a process-once assumption that skips
  re-registration against a fresh env), and its teardown calls the EP's
  `RyzenAI_Shutdown` out-of-band (Windows-only, no unregister). Recreation
  requires reworking the ctor to key off current-env registration and making
  setup/teardown symmetric, gated on the EP tolerating register → shutdown →
  register (§15 q3). Changing how it is registered may affect current callers,
  so this stage carries its own compatibility review.
- **VitisAI.** Migrate when there is a concrete use case; leave the stub
  until then.

Non-goals:

- **Multi-device-per-EP** (multi-GPU). §6 enumerates all matching
  `OrtEpDevice` entries and de-duplicates their mem-infos, but *choosing which
  device-local allocator to run a model on* when several distinct devices are
  present is a separate problem.

## 14. Delivery stages

Each stage is independently shippable and testable.

**Stage 0 — re-init foundation (EP-agnostic, unblocks Foundry Local).**
Implement the §3 ownership refactor: `OrtGlobals` owns the interfaces
(get-or-create, no `call_once`), `GetOrtGlobals()` re-creates when null,
`OgaShutdown` unloads the add-on libraries via `~OrtGlobals` (moving the
`GetCudaInterface` `LibraryHandle` + interface pointer into `OrtGlobals`), and
the hard-coded `RyzenAIInterface::Shutdown()` goes away. Make CPU `InitOrt`
idempotent (§9). Foundry Local then unloads its last model, calls
`OgaShutdown`, optionally unregisters its EPs on its own env reference, then
`ReleaseEnv`. Tests: shutdown → re-init cycle (CPU first; each EP added as it
migrates).

**Stage 1 — shared-allocator plumbing (EP-agnostic).**

- Add the two ORT wrappers (§4): `Ort::GetMemoryInfo(const OrtEpDevice*,
  OrtDeviceMemoryType)` and `Ort::GetSharedAllocator(OrtEnv*, const
  OrtMemoryInfo*)`.
- Add the §6 `GetEpDevices`-based mode lookup + `GetHostAccessibleAllocator`
  to `DeviceInterface`.
- `OgaUnregisterExecutionProviderLibrary` forwards to ORT's unregister
  (§8, subject to the usage-finished rule); document the §8 ordering rules and
  copy the §5 lifetime contract into the public C API header.

**Stage 2 — WebGPU.** Fetch-on-demand shared allocator; `Model::Model`
branches on the §6 check so WebGPU skips `EnsureDeviceOrtInit` (§12.1).

**Stage 3 — CUDA / NvTensorRtRtx.** Remove the static allocator, add the
host-accessible allocator path, keep the provider-bridge path; add-on library
teardown via the Stage 0 `OrtGlobals` ownership (§12.2).

**Stage 4 — remaining EPs, one at a time.** QNN and OpenVINO, then DML (move
off the trivial-session path), then RyzenAI (ctor/teardown rework + symmetric
register/unregister + reload proof), and VitisAI when it has a use case
(§13).

## 15. Open questions / to verify

1. **genai add-on library unload/reload safety** — verify that unloading the
   genai CUDA add-on library at `OgaShutdown` and reloading it on the next
   re-init is safe w.r.t. the CUDA runtime. This is genai-side only; ORT EP
   libraries are handled by env teardown (§3). If unload/reload proves
   problematic, keep the add-on resident and recreate only the interface (the
   §3 future optimization).
2. Minimum ORT version shipping `EpDevice_MemoryInfo` and `GetSharedAllocator`
   vs the version genai already requires for plugin EP loading. (API names
   confirmed in `onnxruntime_cxx_api.h`.)
3. **RyzenAI EP re-init** — does the RyzenAI EP tolerate register →
   `RyzenAI_Shutdown` → register within one process? Gates the RyzenAI stage
   (§14 stage 4); if not, RyzenAI stays a documented not-recreatable
   exception.
