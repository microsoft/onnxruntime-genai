'use strict';

const path = require('node:path');
const fs = require('node:fs');

const configurations = [
  process.env.ORTGENAI_NODE_CONFIG,
  'Release',
  'RelWithDebInfo',
  'Debug',
  'MinSizeRel',
].filter((value, index, values) => value && values.indexOf(value) === index);
const candidates = configurations
  .map((configuration) =>
    path.join(__dirname, 'build', configuration, 'onnxruntime_genai_node.node'),
  )
  .concat(path.join(__dirname, 'build', 'onnxruntime_genai_node.node'));

let binding;
let lastError;
for (const candidate of candidates) {
  if (!fs.existsSync(candidate)) continue;
  try {
    binding = require(candidate);
    break;
  } catch (error) {
    lastError = error;
  }
}
if (!binding) {
  throw new Error(
    'The ONNX Runtime GenAI Node addon could not be loaded. Build it as described in README.md ' +
      'and ensure its native dependencies are discoverable. ' +
      `Last load error: ${lastError ? lastError.message : 'no addon binary was found'}`,
  );
}

module.exports = binding;
