# Plan: Use plugin-EP shared allocators instead of dummy `OrtSession`

Status: Draft for review. Scope is the WebGPU and CUDA `DeviceInterface`s only;
the same shape will be applied to QNN / NvTensorRtRtx / RyzenAI / etc. in later
work. DML is explicitly out of scope and stays on its current path.

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
session is no longer needed and `OgaUnregisterExecutionProviderLibrary` can be
made to clean up properly.

## 2. Goal

For EPs that genai owns (WebGPU and CUDA initially):

1. When the EP is registered via `OgaRegisterExecutionProviderLibrary`, look up
   the `OrtEpDevice` and the device's `OrtMemoryInfo`, retrieve the env's
   shared allocator, and hand it to the matching `DeviceInterface`.
2. When the EP is unregistered, the `DeviceInterface` drops every reference it
   holds to the EP (allocator, memory info, stream, EP-specific objects), and
   any cached graph sessions targeting the EP are purged before
   `Ort::UnregisterExecutionProviderLibrary` runs.
3. Keep the existing trivial-session path available for code paths that don't
   go through the plugin API (today: the non-plugin / provider-bridge CUDA
   path).
4. No behavioral change for DML, QNN, OpenVINO, RyzenAI, NvTensorRtRtx in this
   change.

## 3. ORT API surface we actually need

Per current ORT design the shared allocators are added to the env at
`RegisterExecutionProviderLibrary` time and removed at
`UnregisterExecutionProviderLibrary` time. We do **not** need to call
`CreateSharedAllocator` / `ReleaseSharedAllocator` from genai (they exist on
`Ort::Env` in the upstream C++ API but the env-managed registration covers our
use). We only need to:

- Get the `OrtEpDevice` list for a given EP name — already wrapped as
  `Ort::GetEpDevices` in [`onnxruntime_inline.h`](../src/models/onnxruntime_inline.h#L138).
- Get an `OrtMemoryInfo` from an `OrtEpDevice` — **new thin wrapper**
  matching the upstream C++ API `Ort::EpDevice::GetMemoryInfo(OrtDeviceMemoryType)`
  (wraps `OrtApi::EpDevice_MemoryInfo`). Called once for
  `OrtDeviceMemoryType_DEFAULT` (device-local) and once for
  `OrtDeviceMemoryType_HOST_ACCESSIBLE` when the EP advertises pinned host
  memory. Returns a non-owning `const OrtMemoryInfo*`.
- Get the env's shared allocator for a mem-info — **new thin wrapper**
  matching the upstream `Ort::Env::GetSharedAllocator(const OrtMemoryInfo*)`
  (wraps `OrtApi::GetSharedAllocator`). Returns a non-owning `Ort::Allocator*`;
  the env owns it.

So the wrapper additions are limited to two small inline functions, not the
full Create/Release set. Upstream signatures (from
`include/onnxruntime/core/session/onnxruntime_cxx_api.h`) for reference:

```cpp
// On Ort::EpDevice (via EpDeviceImpl)
ConstMemoryInfo GetMemoryInfo(OrtDeviceMemoryType memory_type) const;

// On Ort::Env
UnownedAllocator GetSharedAllocator(const OrtMemoryInfo* mem_info);
```

The genai-side wrappers will follow the existing genai `Ort::` namespace style
(free functions taking raw C handles) rather than the upstream wrapper types.

## 4. Lifetime contract (documented, not enforced in v1)

> No `Model`, `Generator`, `Tokenizer`, `OgaTensor`, `Engine`, or any object
> that holds a `DeviceBuffer` allocated from an EP may outlive the
> corresponding `OgaUnregisterExecutionProviderLibrary` call.
>
> The caller is responsible for destroying every such object before
> unregistering the EP. Unregistering an EP with live device buffers is
> undefined behavior (typically a crash inside the buffer's destructor when
> it tries to free through a now-invalid allocator).

This is the same contract DML already relies on today via the
`Model::~Model` → `CloseDmlInterface` block in
[`model.cpp`](../src/models/model.cpp#L603); we are just generalizing and
documenting it. The unregister path will assert (debug builds) that no
buffers are outstanding for the EP, but no production refcount/sharing of
the allocator is introduced.

## 5. `DeviceInterface` changes

Source: [`smartptrs.h`](../src/smartptrs.h#L101).

- Keep `InitOrt(const OrtApi&, Ort::Allocator&)` for the non-plugin
  bootstrap path (CUDA provider-bridge, DML, anything still going through
  `EnsureDeviceOrtInit`).
- Add two new virtuals with default no-op implementations:
  - `virtual void OnEpRegistered(OrtEnv& env, const OrtEpDevice& device)`
  - `virtual void OnEpUnregistering(OrtEnv& env) noexcept`
- Add `virtual Ort::Allocator* GetHostAccessibleAllocator() { return nullptr; }`
  for EPs that expose a pinned / mappable host allocator (CUDA host-pinned
  for staging). Returns null if not applicable.

Move the cached `Ort::Allocator*` (and any cached `OrtMemoryInfo*`) out of
file-static globals and into the `InterfaceImpl` instance. This eliminates
the existing `assert(!ort_allocator_)` regression on re-init and is required
to support register → unregister → register without a process restart.

`DeviceBuffer` keeps its current shape: it captures the allocator pointer at
construction and calls `Free` in its dtor. Combined with the documented
lifetime contract above this is sufficient — no refcounting of the
`Ort::Allocator` is introduced.

## 6. Registration / unregistration plumbing

In [`ort_genai_c.cpp`](../src/ort_genai_c.cpp#L1116).

### 6.0 Matching `OrtEpDevice` to `DeviceInterface`

The string passed to `OgaRegisterExecutionProviderLibrary` is a
*registration name* — a library-scoped handle ORT uses to identify what to
load and later unload. It is not the EP name. A single library can
register zero, one, or several EPs, each with its own `EpName()` (OpenVINO,
for example, exposes more than one), and the registration name does not
have to match any of them. The only authoritative identifier per
`OrtEpDevice` is `Ort::EpDevice::EpName()`.

Matching therefore happens in two steps:

1. A small static EP-name → `DeviceType` registry inside genai (initially:
   `"WebGpuExecutionProvider"` → `WEBGPU`,
   `"CUDAExecutionProvider"` → `CUDA`,
   `"NvTensorRTRTXExecutionProvider"` → `NvTensorRtRtx`). EPs whose name
   is not in the registry are ignored.
2. Snapshot/diff against `Ort::GetEpDevices(env)` to associate the
   newly-added or about-to-be-removed devices with a given registration
   call. ORT does not expose a "devices added by registration X" query,
   so genai records this association itself.

The association is held inside `OrtGlobals` as something like:

```cpp
std::unordered_map<std::string, std::vector<const OrtEpDevice*>> bound_devices_by_registration_;
```

Entries are added on register and removed on unregister. Pre-existing
EPs picked up by the env-ctor call site (§6.3) are not recorded — they
cannot be unregistered through `Oga*` per the supported pattern.

### 6.1 `OgaRegisterExecutionProviderLibrary(registration_name, path)`

1. Snapshot the current `Ort::GetEpDevices(env)` set as `before`.
2. `Ort::RegisterExecutionProviderLibrary(env, registration_name, path)`
   (as today).
3. Snapshot `Ort::GetEpDevices(env)` as `after`. `new_devices = after - before`.
4. For each device in `new_devices`:
   - Look up `device.EpName()` in the registry. If unknown, skip
     (forward-compatible).
   - Call `GetDeviceInterface(type)->OnEpRegistered(env, *device)` unless
     the helper's per-device or per-`DeviceInterface` guard (§6.3)
     short-circuits it.
   - Record `device` under `bound_devices_by_registration_[registration_name]`.

### 6.2 `OgaUnregisterExecutionProviderLibrary(registration_name)`

1. Look up `bound_devices_by_registration_[registration_name]`. If absent,
   forward to ORT and return (host registered a library genai doesn't own;
   nothing for us to clean up).
2. For each device in that list, in some deterministic order:
   - `GetDeviceInterface(type_for(device.EpName()))->OnEpUnregistering(env)`.
     This must drop cached `Ort::Allocator*`, `OrtMemoryInfo*`,
     `OrtSyncStream`, and any EP-specific state.
   - (Debug) assert that no `DeviceBuffer`s are outstanding for the EP.
3. Purge entries in
   [`OrtGlobals::graph_session_cache_`](../src/generators.h#L160) that
   target any of those EPs. (Add a small `ep_name` field to the cache
   key.)
4. Erase `bound_devices_by_registration_[registration_name]`.
5. `Ort::UnregisterExecutionProviderLibrary(env, registration_name)`.

Note that genai's cleanup must happen *before* ORT's library unload —
flipping the order would invalidate the allocators while genai still
holds pointers to them.

### 6.3 Externally-registered or pre-existing EPs

An EP doesn't have to enter the env through `OgaRegisterExecutionProviderLibrary`.
The situation is sharpened by the fact that `OrtEnv` is a process-wide
singleton: the first caller to invoke `OrtCreateEnv` (or its C++ wrapper)
wins, and every subsequent call returns the same instance. genai's
[`OrtGlobals::OrtGlobals()`](../src/generators.cpp#L65) calls
`OrtEnv::Create(...)` exactly once, but the host application may have
beaten it to it.

**Supported pattern:** any plugin EP that the host app wants to register
directly against the `OrtEnv` (via `OrtApi::RegisterExecutionProviderLibrary`)
must be registered **before** the first genai API call — i.e. before the
function-static `OrtGlobals` inside
[`GetOrtGlobals()`](../src/generators.cpp#L86) is materialised. After that
point, EP registration must go through `OgaRegisterExecutionProviderLibrary`
so genai can bind the shared allocator to the matching `DeviceInterface`.

This constraint keeps the design small: there are exactly two windows where
an EP can become visible to genai, and both have a single, well-defined
hook.

Cases the design has to handle:

- **Host pre-registered the env:** Host creates the singleton `OrtEnv` and
  registers plugin EPs against it before the first genai API call (i.e.
  before `GetOrtGlobals()` materialises `OrtGlobals` and runs
  `OrtEnv::Create`). genai's `OrtEnv::Create` adopts the existing env;
  `OgaRegisterExecutionProviderLibrary` is never called for those EPs.
- **genai wrapper path:** Host calls `OgaRegisterExecutionProviderLibrary`
  after genai is initialised. Standard case.
- **Built-in EPs:** CPU (and any other statically linked EP) is always
  present. Out of scope for shared-allocator binding because CPU doesn't
  go through this path, but the discovery walk needs to ignore it gracefully.

Factor the bind step out of `OgaRegisterExecutionProviderLibrary` into a
helper so both entry points share it:

```cpp
// Walks the given devices, looks each EP name up in the registry (§6.0), and dispatches OnEpRegistered for
// any genai DeviceInterface that doesn't yet have an allocator bound. Records the bound devices under
// registration_name in OrtGlobals; pass empty for pre-existing devices that won't be unregistered through Oga*.
void BindSharedAllocators(OrtEnv& env,
                          std::span<const OrtEpDevice* const> devices,
                          std::string_view registration_name);
```

Call sites — each runs at most a handful of times per process, never on a
hot path:

1. End of `OrtGlobals::OrtGlobals()` in
   [`generators.cpp`](../src/generators.cpp#L65), after `env_` is
   constructed and the CPU allocator is registered. Passes the full
   `Ort::GetEpDevices(env)` list with an empty registration name. Runs
   exactly once (the ctor is gated by the function-static
   `std::make_unique<OrtGlobals>()` inside `GetOrtGlobals()`). Picks up
   any EP the host app pre-registered against the env singleton.
2. Inside `OgaRegisterExecutionProviderLibrary` (§6.1 step 4), passing
   the snapshot/diff `new_devices` and the registration name. Replaces
   the inline walk that used to live in §6.

The helper itself owns the idempotency, so `OnEpRegistered` implementers
don't have to. Two cheap guards inside the helper:

- Per `DeviceInterface`: skip the dispatch when `HasSharedAllocator()` is
  already true. Handles the "a single library exposes several
  `OrtEpDevice`s that all map to the same `DeviceType`" case (multi-GPU
  with one CUDA EP, etc.) — first device wins, the rest are recorded but
  not re-dispatched.
- Per `OrtEpDevice`: a small `std::unordered_set<const OrtEpDevice*>`
  inside `OrtGlobals` of devices we've already dispatched. Cleared on
  `OnEpUnregistering` for each device in the matching registration entry,
  so a register → unregister → register cycle works.

With those two guards, `OnEpRegistered` is called exactly once per
`(DeviceInterface, OrtEpDevice)` pair across the program's lifetime
between register/unregister boundaries, and implementations can assume
"first call wins, allocator slots are empty".

Unregister side:

- `OgaUnregisterExecutionProviderLibrary` runs `OnEpUnregistering` and
  forwards to `Ort::UnregisterExecutionProviderLibrary`.
- Per the supported pattern above, callers must not invoke
  `OrtApi::UnregisterExecutionProviderLibrary` directly against the env
  while genai is up. If they do, genai's cached allocator dangles —
  documented as unsupported.
- For EPs that were pre-registered by the host and never registered
  through `Oga*`, there's nothing for genai to clean up beyond clearing
  the per-EP entry in the seen-devices set on shutdown. Env teardown frees
  the EP itself.

## 7. `EnsureDeviceOrtInit` and `OrtGlobals::device_allocators_`

Keep both, narrowed in purpose:

- The plugin path (WebGPU always, CUDA when the user registered the CUDA EP
  via the plugin API) bypasses `EnsureDeviceOrtInit` entirely and uses the
  shared allocator stashed by `OnEpRegistered`.
- The non-plugin path (CUDA provider-bridge via
  `AppendExecutionProvider_CUDA_V2` in
  [`cuda/session_options.cpp`](../src/cuda/session_options.cpp#L66))
  continues to call `EnsureDeviceOrtInit` and populate
  `OrtGlobals::device_allocators_[CUDA]`. Removing that requires the CUDA
  provider-bridge code path to go away, which is a separate decision.

`Model` construction needs to be able to use either source. Proposed shape
in [`model.cpp`](../src/models/model.cpp#L580):

```cpp
if (p_device_->HasSharedAllocator()) {
  // Plugin EP path: allocator was set up at register time.
} else {
  EnsureDeviceOrtInit(*p_device_, *config_);  // legacy path
}
```

`HasSharedAllocator()` is a trivial accessor on `DeviceInterface` that
returns `device_allocator_ != nullptr`. `EnsureDeviceOrtInit` keeps its
current "early-return if already populated" behavior so it's safe to call
either order.

Once CUDA's non-plugin path is retired, both `EnsureDeviceOrtInit` and
`OrtGlobals::device_allocators_` can be deleted. Until then they stay.

## 8. Host-accessible allocator

Audit of current code:

- **WebGPU.** `WebGPUMemory::AllocateCpu` `malloc`s a staging buffer and
  the device↔CPU copies go through `OrtEnv::CopyTensors`. If/when the
  WebGPU EP exposes a host-mappable mem-info we should adopt it; until then
  we keep the malloc path. No host allocator needed in v1.
- **CUDA.** [`GpuMemory::AllocateCpu`](../src/cuda/interface.cpp#L46) calls
  `::cudaHostAlloc` directly. The CUDA EP exposes a pinned host allocator —
  if we can pick it up via `EpDevice_MemoryInfo(..., HostAccessible)` +
  `GetSharedAllocator`, we use it. If not, the `cudaHostAlloc` fallback
  stays. This is the only EP in scope that needs a second allocator in v1.
- **DML / QNN / RyzenAI / OpenVINO.** Out of scope; current behavior
  retained.

`DeviceInterface::GetHostAccessibleAllocator()` returns the second allocator
when available, `nullptr` otherwise. Code that uses it (initially: the CUDA
`GpuMemory` ctor / dtor) falls back to `cudaHostAlloc` / `cudaFreeHost` when
it gets `nullptr`.

## 9. Per-EP impact (in scope)

### 9.1 WebGPU

Files: [`webgpu/interface.cpp`](../src/webgpu/interface.cpp),
[`webgpu/session_options.cpp`](../src/webgpu/session_options.cpp).

- `InterfaceImpl::OnEpRegistered`: look up the WebGPU `OrtEpDevice` and its
  device-local mem-info, retrieve the shared allocator, cache both on the
  instance.
- `InterfaceImpl::OnEpUnregistering`: clear `ort_allocator_`,
  `ort_memory_info_`, and `mask_staging_buffer_i32_/i64_`.
- Drop the `assert(!ort_allocator_)` in `InitOrt`. `InitOrt` itself can
  stay as a no-op (or be made unreachable for WebGPU) — the plugin path is
  the only WebGPU path.
- Delete the WebGPU branches in `EnsureDeviceOrtInit` (the trivial-session
  options forwarding and the `WebGPU_Buf` / `WebGPU_Buffer` fallback) once
  no callers remain.

### 9.2 CUDA / NvTensorRtRtx

Files: [`cuda/interface.cpp`](../src/cuda/interface.cpp),
[`cuda/session_options.cpp`](../src/cuda/session_options.cpp),
[`nvtensorrtrtx/session_options.cpp`](../src/nvtensorrtrtx/session_options.cpp).

- Move the file-static `ort_allocator_` and `device_label` into the
  `CudaInterfaceImplBase` instance so the interface can hold per-EP state
  again. Update `GpuMemory` to capture the allocator pointer at
  construction (e.g. store it on the buffer or look it up via the
  interface) instead of reading the namespace global.
- `OnEpRegistered`: same shape as WebGPU. Also probe the EP device for a
  host-accessible mem-info; if present, populate
  `host_accessible_allocator_` and use it from `GpuMemory::AllocateCpu` /
  `~GpuMemory`.
- `OnEpUnregistering`: clear cached allocators / mem-info. Do **not**
  destroy `g_stream` yet — the stream is genai-owned and re-used across
  registrations today; teardown stays at `Shutdown()` for now.
- Keep `InitOrt` working as today for the provider-bridge path. The two
  paths are mutually exclusive at runtime: if `OnEpRegistered` ran first,
  `InitOrt` is a no-op for CUDA (skip when `device_allocator_` is already
  set); if `InitOrt` ran first, `OnEpRegistered` will never be called
  because the plugin EP wasn't registered.
- `cuda/session_options.cpp::AppendExecutionProvider` keeps its current
  "try V2 plugin, fall back to provider-bridge" logic. No change there.

## 10. Out of scope for this change

- **DML.** Stays on the current trivial-session + `g_dml_device` +
  `CloseDmlInterface` path. The plugin shape will be added later when /
  if DML moves to a plugin EP.
- **QNN, OpenVINO.** No change; still use the trivial-session path. Will
  follow once they move to plugin-only registration.
- **RyzenAI.** Out of scope to avoid a breaking change. Today
  `GetRyzenAIInterface()` self-registers the EP on first call (with
  fallback search for `onnxruntime_providers_ryzenai.dll` next to
  `onnxruntime-genai.dll`, next to `onnxruntime.dll`, next to the current
  executable, or in CWD; see
  [`ryzenai/interface.cpp`](../src/ryzenai/interface.cpp)). The genai
  binary is built with RyzenAI sources unconditionally — there is no
  `USE_RYZENAI` flag, the files are pulled in by the cmake glob in
  [`global_variables.cmake`](../cmake/global_variables.cmake#L78). Users
  do not call `OgaRegisterExecutionProviderLibrary` for RyzenAI today.
  Moving RyzenAI to require explicit registration would break every
  current caller; that migration needs its own RFC / deprecation window.
- **VitisAI.** Leave the existing stub as-is until there is a concrete
  use case.
- **Multi-device-per-EP** (multi-GPU). The new hooks pass a single
  `OrtEpDevice`; if multiple devices match the EP name, v1 picks the
  first. Selecting between them is a separate problem.

## 11. Sequencing

1. Add the two `OrtApi` wrappers in
   [`onnxruntime_api.h`](../src/models/onnxruntime_api.h) /
   [`onnxruntime_inline.h`](../src/models/onnxruntime_inline.h):
   - `Ort::GetMemoryInfo(const OrtEpDevice*, OrtDeviceMemoryType)` →
     `const OrtMemoryInfo*` (wraps `OrtApi::EpDevice_MemoryInfo`).
   - `Ort::GetSharedAllocator(OrtEnv*, const OrtMemoryInfo*)` →
     `Ort::Allocator*` non-owning (wraps `OrtApi::GetSharedAllocator`).
2. Add `OnEpRegistered` / `OnEpUnregistering` /
   `GetHostAccessibleAllocator` / `HasSharedAllocator` to
   [`DeviceInterface`](../src/smartptrs.h#L101) with default
   implementations. No call sites yet.
3. Add the `BindSharedAllocators` helper (§6.3) and the
   `bound_devices_by_registration_` map on `OrtGlobals`. Call the helper
   from:
   - End of `OrtGlobals::OrtGlobals()` after `env_` is constructed, with
     the full `GetEpDevices` list and an empty registration name.
   - `OgaRegisterExecutionProviderLibrary` after the underlying ORT
     register call, with the snapshot/diff `new_devices` and the
     registration name (§6.1).

   Wire `OgaUnregisterExecutionProviderLibrary` per §6.2: look up the
   per-registration device list, call `OnEpUnregistering` for each
   bound `DeviceInterface`, purge `graph_session_cache_`, erase the map
   entry, then forward to `Ort::UnregisterExecutionProviderLibrary`.
4. Migrate WebGPU (smallest blast radius, no kernels, no host-accessible
   allocator needed).
5. Update `Model::Model` to branch on `HasSharedAllocator()` so WebGPU can
   skip `EnsureDeviceOrtInit`.
6. Migrate CUDA / NvTensorRtRtx (move statics into the instance, add the
   host-accessible allocator path, keep the provider-bridge path working).
7. Documentation: copy the lifetime contract from §4 into the public C API
   header next to `OgaRegisterExecutionProviderLibrary` /
   `OgaUnregisterExecutionProviderLibrary`.

## 12. Open questions

1. Confirm the minimum ORT version that ships `EpDevice_MemoryInfo` and
   `GetSharedAllocator` against the version genai already requires for
   plugin EP loading. (API names are confirmed:
   `Ort::EpDevice::GetMemoryInfo` and `Ort::Env::GetSharedAllocator` in
   `onnxruntime_cxx_api.h`.)
2. Confirm that the CUDA EP plugin advertises a host-accessible mem-info
   for pinned memory (i.e. `EpDevice::GetMemoryInfo(OrtDeviceMemoryType_HOST_ACCESSIBLE)`
   returns a valid mem-info). If not, the host allocator change is a no-op
   for v1 and `cudaHostAlloc` stays.
3. Should `OgaUnregisterExecutionProviderLibrary` for an EP that genai
   doesn't recognize still forward to ORT? (Yes — needed for caller-owned
   EPs.) Should it noisily ignore an unregister for an EP that still has
   outstanding buffers, or fail hard? (Recommend: assert in debug, log +
   forward in release, document UB.)
