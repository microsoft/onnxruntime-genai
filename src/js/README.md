# ONNX Runtime GenAI non-generative Node binding

This package is a synchronous Node-API binding for the stable non-generative C
ABI. It exposes directory tokenization and typed CLM/KEV package sessions. It
does not expose generation APIs or duplicate package preprocessing.

## Build

Install the JavaScript build dependency, then build either in the main CMake
tree with `-DENABLE_JAVASCRIPT=ON`, or against an installed core:

```sh
npm install --no-package-lock --prefix src/js
cmake -S src/js -B src/js/build \
  -Donnxruntime-genai_DIR=/path/to/genai/lib/cmake/onnxruntime-genai \
  -DORT_HOME=/path/to/onnxruntime
cmake --build src/js/build --config Release
npm test --prefix src/js
```

The build copies the matching GenAI core, optional CUDA companion, and ONNX
Runtime/provider libraries beside the addon. Linux and macOS use an
origin-relative loader path, so an installed npm package does not depend on the
build machine's library paths. GPU driver/toolkit libraries remain system
dependencies.
Windows builds also require the Node import library; pass
`-DNODE_LIBRARY=C:\path\to\node.lib` if CMake cannot find it beside Node.

`npm pack` generates a platform package named
`onnxruntime-genai-non-generative-<os>-<arch>` with matching npm `os` and `cpu`
restrictions. Publish each supported build separately; native artifacts are not
cross-platform. The loader searches Release, RelWithDebInfo, Debug, and
MinSizeRel outputs. Set `ORTGENAI_NODE_CONFIG` to select a configuration first.

## API and ownership

`DirectoryTokenizer`, `RankingSession`, and `DecisionSession` own their native
handles. `close()` is idempotent; every other operation throws after close.
Native calls and close are serialized per object. Methods are synchronous.

Structured values are ordinary JavaScript values. Integral `number` values
must be within JavaScript's safe integer range; use `bigint` for the full signed
64-bit range. Non-integral finite numbers use native doubles. Objects are
traversed in JavaScript own-key order and the native ordered object builder is
used. Native int64 results outside the safe range are returned as `bigint`.
Cycles are rejected and nesting is limited to 128 containers; repeated
references in separate non-cyclic branches remain valid.

```js
const packageName =
  `onnxruntime-genai-non-generative-${process.platform}-${process.arch}`;
const { RankingSession } = require(packageName);
const usingSession = new RankingSession('/models/clm', ['cpu']);
const result = usingSession.run({
  state: { weather: 'rain' },
  questions: { umbrella: { type: 'noul', instructions: 'Take one?' } },
});
usingSession.close();
```

Set `ORTGENAI_TEST_CLM_PACKAGE` and/or `ORTGENAI_TEST_KEV_PACKAGE` to enable
real-package integration tests.
