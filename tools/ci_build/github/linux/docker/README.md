# Refreshing GenAI build images

The base images are produced and promoted upstream, not by this repository.
[`base-images.json`](base-images.json) centralizes the consumed repositories,
immutable digests, provenance tags/builds, and Dockerfile default families.
Tags are provenance metadata; consumers use `repository@digest`.

The AlmaLinux8 entries use promoted official images. Producer build and source
commit provenance are recorded in the manifest. The pins match
[microsoft/onnxruntime#33777](https://github.com/microsoft/onnxruntime/pull/33777)
at commit `898786027ebd16e53f3db80d4b8a96f41749b2aa`, but the default map covers
GenAI's own six Dockerfiles and only the families it consumes:

| Consumer | Base family |
| --- | --- |
| manylinux CPU x64 | AlmaLinux8 CPU x64 |
| manylinux CPU ARM64 | AlmaLinux8 CPU ARM64 |
| manylinux CUDA 12.8 x64 | AlmaLinux8 CUDA12 x64 |
| manylinux CUDA 13 x64 | AlmaLinux8 CUDA13 x64 |
| manylinux CUDA 13 ARM64 | Historical AlmaLinux9 CUDA13 ARM64 |
| inference CPU ARM64 | AlmaLinux8 CPU ARM64 with dotnet |
| ADO CUDA integration tests and engine benchmarks | AlmaLinux8 CUDA12 x64 |

CUDA13 ARM64 retains its `20260323.1` AlmaLinux9 image, pinned by digest.
Replacing it with AlmaLinux8 would change the manylinux/glibc ABI. Refreshing
that historical family remains separate work.

## Resolution and cache behavior

`tools/ci_build/get_docker_image.py` resolves the Dockerfile default before
building. Both ADO packaging/integration builds and the GitHub CUDA x64,
CUDA ARM64 and CPU ARM64 workflows already use this helper. GenAI has no
consumer of the upstream GHCR-caching `build-docker-image` action, so it does
not need ORT's wrapper action. If such a consumer is added, resolve the base
into its build arguments **before** that action calculates its cache key.
The resolver's `--github-output` option can emit those resolved arguments.

ADO integration tests and engine benchmarks resolve the same CUDA12 family
before `docker run`, rather than retaining separate image pins.
`integration-pytest-step.yml` accepts `cudaBaseImageFamily`; its existing
`cudaDockerImage` parameter remains available for explicit custom images.

For routine refreshes, change only the affected `images` entries and associated
`producer_builds` provenance in the manifest after verifying producer promotion.
`--base-image-family` selects a non-default family; an explicit
`--build-arg BASEIMAGE=<reference>` is also supported. Unknown families,
duplicate/empty explicit base arguments, and simultaneous family/explicit-image
selections fail. Custom bases must preserve the consumer's architecture, OS/ABI,
CUDA and dotnet requirements and provide Node/npm where inherited.

The helper preserves existing registry hosts, derived image names, publishing
and layer-cache behavior, and requests base resolution with `--pull` in every
build mode. A changed immutable digest invalidates the dependent Docker layers.
For an authorized rebuild of package-install layers without a base change,
pass `--no-cache` in `--docker-build-args`. The ADO
`pull-manylinux-image.yml` template forwards `base_image_family` and
`docker_build_args` overrides.

From the repository root, a local non-publishing build can use:

```shell
python tools/ci_build/get_docker_image.py \
  --dockerfile tools/ci_build/github/linux/docker/manylinux/Dockerfile.manylinux2_28_cuda_12.8 \
  --context tools/ci_build/github/linux/docker/manylinux \
  --repository genai-cuda-local \
  --docker-build-args="--build-arg BUILD_UID=1001"
```

Omit `--container-registry` to avoid publishing. Managed Dockerfiles declare
`ARG BASEIMAGE` without a default because Docker cannot read the manifest.
Raw `docker build` commands must supply `--build-arg BASEIMAGE=<reference>`.
Some Dockerfiles also require the existing `pip_index_url` BuildKit secret.

## Inherited tooling and validation

The refreshed AlmaLinux8 bases supply Node 24.22.2 and npm 11.21.1. The ARM64
CPU inference dependency script no longer overwrites them with Node 18.
Manylinux dependency scripts have no duplicate Node installation to remove;
there is no GenAI `install_shared_deps.sh` or OpenVINO Dockerfile cleanup.
The historical AlmaLinux9 family keeps its existing inherited tools.

The new producer interpreters are CPython 3.11--3.14, not 3.10. GenAI's
manylinux dependency installer and SDK wheel matrix already select 3.11--3.14;
core builds select 3.11 and CUDA integration build/test/benchmark paths select
3.12. No CPython 3.10 migration is needed for these consumers.

Rebuild and scan the derived images and run their architecture-specific
build/test gates before release. A repin is not a clean vulnerability scan:
npm 11.21.1 bundles `ip-address` 10.7.2, below the report target of 10.7.4,
and retained Nsight `nic_sampler` requires a patched NVIDIA payload rather
than a host Go upgrade. Keep profiling enabled and track these remediation
gates separately. Do not manually trigger ADO publishing jobs or overwrite
shared derived-image `:latest` tags merely to validate configuration.
