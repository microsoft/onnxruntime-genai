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
5. No behavioral change for DML / QNN / OpenVINO / RyzenAI beyond what is
   stated here. (NvTensorRtRtx now uses the plugin shared-allocator path — see
   §12.2 / Stage 3 — as it is a plugin-only EP.)

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
- **`~OrtGlobals` tears down in the reverse of construction order:** everything
  that uses env-owned resources (shared allocators, registered EP libraries) is
  destroyed **before** the env. Concretely: session caches → device interfaces
  (`owned_interfaces_`) + CUDA add-on library (`cuda_library_`) → legacy
  trivial-session allocators (`device_allocators_`) → `env_`. This ends all
  shared-allocator usage before the env clears them, so it stays correct even if
  a future interface/buffer dereferences a cached allocator in its own
  destructor — it does not rely on "nothing derefs it today". (This inverts an
  earlier draft that dropped `env_` first to preserve RyzenAI's legacy
  post-env shutdown; RyzenAI is updated to fit the clean order instead — §13.)
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
interface's EP name appears in `Ort::GetEpDevices`**. Present (a matching
`OrtEpDevice` exists) → plugin mode; absent → legacy mode
(`EnsureDeviceOrtInit`). The **presence of an `OrtEpDevice` is the plugin-EP
signal**; whether that EP also exposes a usable shared allocator is a
**separate** concern, resolved afterwards. Each interface knows its own EP
name(s) and its own env (in-process EPs call `GetOrtEnv()`; the CUDA add-on
stores the env passed at construction). Detection is a **two-step split**
([`smartptrs.h`](../src/smartptrs.h)):

- **Step 1 — `FindEpDevicesByName(env, ep_names)`**: pure name filtering over
  `GetEpDevices`, returning every matching `OrtEpDevice` (an EP name can appear
  on several entries, e.g. multi-GPU). Non-empty → plugin mode.
- **Step 2 — `ResolveEpSharedAllocators(env, devices)`**: takes that device
  list (independent of name filtering) and resolves the shared allocators,
  returning an `EpSharedAllocators` — the device-local (`DEFAULT`) allocator and
  the optional `HOST_ACCESSIBLE` allocator (only when a matched device is
  non-CPU, §11), each with its mem-info, plus `HasDeviceAllocator()` /
  `HasHostAccessibleAllocator()` predicates. "Availability" is decided by
  whether `GetSharedAllocator` actually returns an allocator, **not** by whether
  a mem-info is advertised.

```cpp
// Step 1: name filter — presence == plugin EP. env is the interface's own.
std::vector<const OrtEpDevice*> FindMyEpDevices() {
  return FindEpDevicesByName(env, ep_names_);  // empty -> legacy mode
}
```

**WebGPU is the sole exception.** ORT registers a built-in ("internal") WebGPU
plugin EP that creates an `OrtEpDevice` but provides **no** shared allocator,
and genai must run that through the legacy trivial-session path rather than
plugin mode. So WebGPU — and only WebGPU — additionally gates `FindMyEpDevices`
on `HasDeviceAllocator()` (step 2), treating "device present but no device
allocator" as legacy. Every other EP uses presence alone. (WebGPU has never
been a provider-bridge EP.)

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
- **Mode + allocators via `GetEpDevices`** (§6): step 1 `FindEpDevicesByName`
  for the mode signal, step 2 `ResolveEpSharedAllocators` for the device-local
  and host-accessible shared allocators; may be memoized per env cycle.
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
`p_device_->FindMyEpDevices()` is non-empty it is plugin mode (shared
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

- Allocator fetched on demand; mode + device-local allocator discovered via the
  two-step `FindEpDevicesByName` / `ResolveEpSharedAllocators` split (§6).
- **WebGPU is the plugin-detection exception** (§6): `FindMyEpDevices` gates on
  `HasDeviceAllocator()`, so the internal WebGPU EP (no shared allocator) falls
  through to the legacy path. `InitOrt` is therefore retained for that legacy
  path and sets `ort_allocator_` / `ort_memory_info_` from the trivial-session
  allocator; `EnsureAllocator()` is a no-op afterwards.
- Keep the WebGPU fallback allocator name in `EnsureDeviceOrtInit` for the
  legacy internal EP.

### 12.2 CUDA / NvTensorRtRtx

Files: [`cuda/interface.cpp`](../src/cuda/interface.cpp),
[`cuda/session_options.cpp`](../src/cuda/session_options.cpp),
[`nvtensorrtrtx/session_options.cpp`](../src/nvtensorrtrtx/session_options.cpp).

- Remove the file-static `ort_allocator_`; fetch on demand in plugin mode.
  `device_label` stays (it identifies the buffer type). `GpuMemory` captures
  the allocator at construction from the interface.
- Discover the host-accessible mem-info (if any) via `GetEpDevices` and use
   its shared allocator for `AllocateCpu` / free (§11); in plugin mode both the
  CUDA and TensorRT-RTX plugin EPs advertise one (`CUDAPinnedAllocator`, added
  via `EpDevice_AddAllocatorInfo` alongside the DEFAULT device mem-info). The
  direct `::cudaHostAlloc` / `cudaFreeHost` calls remain only as the
  legacy-mode (provider-bridge) fallback (CUDA only; TensorRT-RTX is
  plugin-only).
- The CUDA stream and kernels are genai add-on-library state, **not**
  EP-scoped: they are created with the interface and destroyed only when the
  add-on library is torn down at `OgaShutdown` (§3). They are not touched by
  EP registration state.
- Keep `InitOrt` for the provider-bridge path; the two paths are mutually
  exclusive at runtime and selected by the §6 check. TensorRT-RTX has no
  provider-bridge path, so it is always plugin mode in practice.
- **EP name handling (CUDA).** The ORT CUDA *plugin* factory currently reports
  the name `"CudaPluginExecutionProvider"` (`cuda_ep_factory.h`) in a released
  ORT, which differs from the legacy/provider-bridge name
  `"CUDAExecutionProvider"`. genai matches **both** names in both places — the
  interface's `FindMyEpDevices` (plugin allocator-mode detection) and
  `AppendExecutionProviderV2` (plugin-vs-bridge append) — so a CUDA plugin
  registered under either name is detected, and the two decisions always agree.
  Only one name is assumed present at a time (the plugin advertises a single
  device name), so `FindMyEpDevices` returns one device and append uses it.
  TensorRT-RTX has no such discrepancy (`"NvTensorRTRTXExecutionProvider"`
  throughout).
- The add-on library does not link `onnxruntime`, so it has no `OrtGetApiBase`
  of its own and no `GetOrtEnv()`. Both `Ort::api` and the `OrtEnv*` are passed
  into the DLL at `GetInterface` time (new `const OrtApi*` and `OrtEnv*`
  parameters supplied by `LoadCudaInterface`). The env exists before the DLL is
  loaded, so the interface stores it (`env_`, const) — guaranteed non-null —
  rather than lazily caching it from `FindMyEpDevices`. Both are available
  before the plugin-mode allocator fetch runs during `Model` construction, so
  the plugin path never depends on `InitOrt`.
- `cuda/session_options.cpp::AppendExecutionProvider` keeps its "try V2
  plugin, fall back to provider-bridge" logic.

## 13. Per-EP notes and non-goals

All EPs are in scope (staged in §14). Per-EP specifics:

- **WebGPU, CUDA / NvTensorRtRtx.** First migrations; detailed in §12.
- **DML.** Currently on the trivial-session + `g_dml_device` +
  `CloseDmlInterface` path, with a `GetDmlInterface()` free-function accessor
  (the only `GetXInterface()` wrapper still remaining — all others were removed
  in Stage 0 in favour of `GetDeviceInterface(DeviceType::X)`). Migrating DML
  follows the same pattern already applied to every other EP:
  1. Add `CreateDmlInterface()` factory (replacing the `InitDmlInterface` +
     `g_dml_device` static ownership pattern).
  2. Remove `GetDmlInterface()` and `CloseDmlInterface()`; DML teardown folds
     into `~OrtGlobals` (owned via `OrtGlobals::owned_interfaces_`).
  3. Update the `#if USE_DML` arm of `OrtGlobals::GetDeviceInterface` to call
     `CreateDmlInterface()` instead of `GetDmlInterface()`.
  4. Update `dml/session_options.cpp`: replace the `!GetDmlInterface()` guard
     (which triggered lazy `InitDmlInterface`) with `GetDeviceInterface`;
     `InitDmlInterface` is absorbed into `CreateDmlInterface`.
  5. Wire onto the §6 `GetEpDevices`-based shared-allocator path (same as
     WebGPU/CUDA).
- **QNN.** Migrate to the shared-allocator path once exercised as a plugin EP.
- **OpenVINO.** Investigated as a plugin EP and left on its original behavior:
  the OpenVINO plugin EP exposes **no** shared allocator on the env
  (`ResolveEpSharedAllocators(...).HasDeviceAllocator()` is false; its "device"
  memory is host-accessible, and for stateful / `enable_causallm` models the KV
  cache is managed inside the EP). So OpenVINO keeps delegating all allocation
  to the CPU interface (`GetDeviceInterface(DeviceType::CPU)`) and is **not**
  wired onto the §6 shared-allocator path. Only its lifecycle changed (Stage 0:
  `CreateOpenVINOInterface`, owned by `OrtGlobals`). `FindMyEpDevices` stays at
  the default (empty), so `EnsureDeviceOrtInit` early-returns for OpenVINO
  exactly as before, and `InitOrt` remains `assert(false)` (never called).
- **RyzenAI.** `GetRyzenAIInterface()` has been removed (Stage 0 cleanup —
  callers now use `GetDeviceInterface(DeviceType::RyzenAI)`).
  The DLL-path probe that used `GetRyzenAIInterface` as a module-address
  marker now uses `CreateRyzenAIInterface` instead. The deeper recreatability
  work remains: its ctor early-returns when the EP module is already resident
  (a process-once assumption that skips re-registration against a fresh env),
  and its teardown calls `RyzenAI_Shutdown` out-of-band (Windows-only, no
  unregister). **With the reverse-order teardown (§3), the RyzenAI interface is
  now destroyed *before* `env_`**, so its `~Interface()` runs while the EP is
  still registered on the live env (previously it ran post-env, when env
  destruction had already unloaded the EP and the shutdown export was a no-op
  fallback). The clean target design is a **symmetric register/unregister**:
  the ctor registers the EP library on the env, the dtor unregisters it on that
  same env *before* the env drops, replacing the out-of-band `RyzenAI_Shutdown`
  export call. This requires reworking the ctor to key off current-env
  registration and making setup/teardown symmetric, gated on the EP tolerating
  register → unregister → register within one process (§15 q3). Changing how it
  is registered may affect current callers, so this stage carries its own
  compatibility review. Until then the existing defensive `~Interface()` (which
  no-ops when the module is absent) remains, and RyzenAI stays a documented
  not-yet-recreatable exception.
- **VitisAI.** Migrate when there is a concrete use case; leave the stub
  until then.
- **`GetXInterface()` wrappers (all EPs except DML).** Removed in Stage 0.
  Every caller now uses `GetDeviceInterface(DeviceType::X)` directly. The
  only remaining wrapper is `GetDmlInterface()`, which is removed as part of
  the DML Stage 4 migration above.

Non-goals:

- **Multi-device-per-EP** (multi-GPU). §6 enumerates all matching
  `OrtEpDevice` entries and de-duplicates their mem-infos, but *choosing which
  device-local allocator to run a model on* when several distinct devices are
  present is a separate problem.

## 14. Delivery stages

Each stage is independently shippable and testable.

**Stage 0 — re-init foundation (EP-agnostic, unblocks Foundry Local).**
Implement the §3 ownership refactor: `OrtGlobals` owns the interfaces
(get-or-create under a lock, no `call_once`), `GetOrtGlobals()` re-creates
when null (guarded by `g_process_exiting`), `OgaShutdown` unloads the add-on
libraries via `~OrtGlobals` (moving the CUDA `LibraryHandle` + interface
pointer into `OrtGlobals`), and the hard-coded `RyzenAIInterface::Shutdown()`
goes away. Make CPU `InitOrt` idempotent (§9). Remove all `GetXInterface()`
wrappers (CPU / WebGPU / QNN / OpenVINO / RyzenAI) — callers use
`GetDeviceInterface(DeviceType::X)` directly; `GetDmlInterface()` remains
until DML is migrated in Stage 4. Add `CreateXInterface()` factories for each
in-process EP; CUDA stays as `OrtGlobals::LoadCudaInterface` (add-on library
path). Also: VS 2026 (MSVC 19.50+) build compatibility — suppress `C4875`
globally and define `_SILENCE_EXPERIMENTAL_COROUTINE_DEPRECATION_WARNINGS` in
`CMakeLists.txt` for third-party deps that haven't updated to VS 2026 yet.
`GetOrtGlobals()` re-creation is mutex-guarded so concurrent first use (or
first use after a shutdown) cannot double-construct the globals.

*Testing.* A dedicated `reinit_tests` executable — separate from `unit_tests`
because it calls `OgaShutdown` (which resets process-global state, so it must
not share a process with the public-API suite) — drives repeated
shutdown → re-init cycles. It links the genai **object library**
(`onnxruntime-genai-obj`) instead of the shared library, giving white-box
access to `GetDeviceInterface` so it can force each available EP's device
interface to be created and then torn down directly (no model load, so it also
covers EPs that can't run the tiny CPU test model). On Windows it registers
WinML-installed EP packages (`Get-AppxPackage "*.EP.*"`, with the provider DLL
found recursively under each package's install location). `unit_tests` stays on
the public C API via the shared library (opt-in `--winml_eps` /  `--ep_dir` to
register plugin EPs there). The object-library split — the shared library and
the white-box tests both consume `onnxruntime-genai-obj` — is what lets tests
reach internal symbols without exporting them from the shipped DLL purely for
testing; the genai library targets are grouped under a `GenAI` IDE solution
folder.

**Stage 1 — shared-allocator plumbing (EP-agnostic).**

- Add the two ORT wrappers (§4): `Ort::GetMemoryInfo(const OrtEpDevice*,
  OrtDeviceMemoryType)` and `Ort::GetSharedAllocator(OrtEnv*, const
  OrtMemoryInfo*)`.
- Add the §6 `GetEpDevices`-based mode lookup + `GetHostAccessibleAllocator`
  to `DeviceInterface`.
- `OgaUnregisterExecutionProviderLibrary` forwards to ORT's unregister
  (§8, subject to the usage-finished rule); document the §8 ordering rules and
  copy the §5 lifetime contract into the public C API header.

**Stage 2 — WebGPU.** Dual-path support: `FindMyEpDevices` virtual added to `DeviceInterface`
(default: empty); `InterfaceImpl` overrides it. `Model::Model` gates on `FindMyEpDevices` —
plugin mode skips `EnsureDeviceOrtInit`; legacy mode calls it (and then `InitOrt` sets
`ort_allocator_` so `EnsureAllocator()` is a no-op for that path). `EnsureAllocator()` fetches
the env's shared allocator on first use in plugin mode. `InitOrt` restored for the legacy V1
path (sets `ort_allocator_` / `ort_memory_info_` from the trivial-session allocator). The WebGPU
fallback allocator name (`"WebGPU_Buffer"`) is retained in `EnsureDeviceOrtInit` for old ORT.
`WebGPUMemory` constructors take `Ort::Allocator&` / `const OrtMemoryInfo&` (references, not
pointers) to make the non-null contract explicit.

**WebGPU is the plugin-detection exception.** The general rule is presence of a
matching `OrtEpDevice` (§6). WebGPU differs because the internal WebGPU EP
(built into ORT) registers itself in `GetEpDevices` as a fake plugin EP but does
**not** call `EpDevice_AddAllocatorInfo`, so ORT exposes **no** shared allocator
for it (`ResolveEpSharedAllocators(...).HasDeviceAllocator()` is false). The real
plugin WebGPU EP (`onnxruntime/core/providers/webgpu/ep/factory.cc`) does call
`EpDevice_AddAllocatorInfo` with `{WEBGPU_BUFFER, OrtMemoryInfoDeviceType_GPU,
OrtDeviceMemoryType_DEFAULT}`, so it does. `FindMyEpDevices` therefore gates on
`HasDeviceAllocator()` for WebGPU only: device present but no device allocator →
legacy path.

**Stage 3 — CUDA / NvTensorRtRtx.** Dual-path support mirroring WebGPU.
The file-static `ort_allocator_` is removed; `GpuMemory` captures its device
(DEFAULT) allocator and optional pinned (HOST_ACCESSIBLE) allocator at
construction. `CudaInterfaceImplBase` gains `FindMyEpDevices` +
`EnsureAllocator`: `FindMyEpDevices` uses the stored `OrtEnv&` (the add-on
library has no `GetOrtEnv()`) and returns the matching CUDA plugin devices by
name (presence == plugin mode; CUDA has no internal fake-plugin EP, and the
provider-bridge path registers no `OrtEpDevice`);
`EnsureAllocator` fetches the env's shared device allocator on first use and,
if the EP advertises a HOST_ACCESSIBLE mem-info (`CUDAPinnedAllocator`), the
shared pinned allocator for `AllocateCpu`/free (§11). Legacy (provider-bridge)
mode leaves `host_allocator_` null so `AllocateCpu` falls back to
`::cudaHostAlloc`. `InitOrt` is retained for the legacy path and is idempotent
(no assert). Each concrete interface passes its EP name(s) to the base
constructor (stored in `ep_names_`): `{"CUDAExecutionProvider",
"CudaPluginExecutionProvider"}` for CUDA (both are matched — the plugin factory
name differs from the legacy name in a released ORT, and only one is present at
a time) and `{"NvTensorRTRTXExecutionProvider"}` for NvTensorRtRtx. Both are
plugin EPs that advertise DEFAULT + HOST_ACCESSIBLE mem-infos. TensorRT-RTX is
plugin-only (no provider-bridge fallback). `AppendExecutionProviderV2` tries
both CUDA names too, so plugin-append and plugin allocator-mode detection always
agree.

**ORT API + env into the add-on DLL:** the add-on DLL's `Ort::api` and the
`OrtEnv*` are passed at `GetInterface` time (new `const OrtApi* ort_api` and
`OrtEnv* ort_env` parameters, supplied by `OrtGlobals::LoadCudaInterface` from
`Ort::api` and `env_`). The env is created before the DLL is loaded, so it is
guaranteed to exist; the interface stores it (`env_`, const) rather than lazily
caching it from `FindMyEpDevices`. The DLL is loaded during
`Model::CreateSessionOptions()`, before the `FindMyEpDevices()` /
`EnsureAllocator()` calls later in the same `Model` constructor, so plugin-mode
`GetEpDevices`/`GetSharedAllocator` lookups have a valid `Ort::api` and env
without relying on `InitOrt` (which the plugin path skips). The CUDA
stream/kernels remain add-on-library state (§12.2), torn down at `OgaShutdown`
via the Stage 0 `OrtGlobals` ownership. Not runtime-validatable without CUDA
hardware; the code compiles under the same declarations as the main module.

*Tests* (authored, in [`test/c_api_tests.cpp`](../test/c_api_tests.cpp)):
`GreedySearchGptFp32CudaCAPI` exercises the plugin path (asserts the CUDA plugin
device advertises DEFAULT **and** HOST_ACCESSIBLE mem-info and that the env
yields a shared allocator for each), and `GreedySearchGptFp32CudaLegacyCAPI`
exercises the provider-bridge path (asserts no plugin DEFAULT mem-info). Both
match EP device names `"CUDAExecutionProvider"` or `"CudaPluginExecutionProvider"`.
They `GTEST_SKIP()` without an NVIDIA GPU, so they must be run on a CUDA machine:
the plugin test with `--ep_dir <dir with onnxruntime_providers_cuda[_plugin].dll>`,
the legacy test against a CUDA-enabled `onnxruntime.dll` with no `--ep_dir`.

**Stage 4 — remaining EPs, one at a time.** OpenVINO (investigated — stays on
CPU delegation, only its lifecycle changed; see §13), QNN (shared-allocator
path, same pattern as WebGPU, when exercised as a plugin EP), then DML (see §13
for concrete steps: add `CreateDmlInterface()`, remove `GetDmlInterface()` +
`CloseDmlInterface()`, fold `g_dml_device` into
`OrtGlobals::owned_interfaces_`, wire onto §6 detection), then RyzenAI
(ctor/teardown rework + symmetric register/unregister + reload proof; §13 for
detail), and VitisAI when it has a use case (§13).

## 15. Open questions / to verify

1. **genai add-on library unload/reload safety** — verify that unloading the
   genai CUDA add-on library at `OgaShutdown` and reloading it on the next
   re-init is safe w.r.t. the CUDA runtime. This is genai-side only; ORT EP
   libraries are handled by env teardown (§3). If unload/reload proves
   problematic, keep the add-on resident and recreate only the interface (the
   §3 future optimization). The `reinit_tests` cycle exercises exactly this on
   a CUDA machine (it forces the CUDA `DeviceInterface`, which loads/unloads
   the add-on, across cycles); *to be run on real CUDA hardware.*
2. **Minimum ORT version — resolved, no check needed.** `EpDevice_MemoryInfo`
   and `GetSharedAllocator` were added as part of the plugin-EP support itself,
   so an EP cannot appear in `GetEpDevices` (i.e. be detected as a plugin EP)
   in an ORT build that lacks these APIs. The §6 plugin-mode detection is
   therefore self-gating: legacy mode is taken whenever the APIs/EP are absent,
   and the plugin path only runs where the APIs exist. No explicit version
   check is required.
3. **RyzenAI EP re-init** — does the RyzenAI EP tolerate register →
   unregister → register within one process, and unregister/shutdown **before**
   env destruction (the reverse-order teardown, §3)? Gates the RyzenAI stage
   (§14 stage 4); if not, RyzenAI stays a documented not-recreatable
   exception.
