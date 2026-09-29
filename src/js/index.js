'use strict';

const path = require('node:path');
const fs = require('node:fs');

const candidates = [
  path.join(__dirname, 'build', 'Release', 'onnxruntime_genai_node.node'),
  path.join(__dirname, 'build', 'onnxruntime_genai_node.node'),
];

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
