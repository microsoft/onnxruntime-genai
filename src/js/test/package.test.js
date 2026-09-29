'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');

function run(command, args, options = {}) {
  const result = spawnSync(command, args, { encoding: 'utf8', ...options });
  assert.equal(
    result.status,
    0,
    `${command} ${args.join(' ')} failed:\n${result.stdout}\n${result.stderr}`,
  );
  return result;
}

test('packed package bundles and resolves native runtime libraries', () => {
  const packageRoot = path.resolve(__dirname, '..');
  const work = path.join(packageRoot, 'build', 'package-test');
  fs.rmSync(work, { recursive: true, force: true });
  fs.mkdirSync(work, { recursive: true });

  const packed = JSON.parse(
    run('npm', ['pack', '--json', '--pack-destination', work], {
      cwd: packageRoot,
    }).stdout,
  )[0];
  const paths = packed.files.map((file) => file.path);
  assert.ok(paths.includes('build/Release/onnxruntime_genai_node.node'));

  const expected = {
    linux: ['build/Release/libonnxruntime-genai.so', 'build/Release/libonnxruntime.so'],
    darwin: [
      'build/Release/libonnxruntime-genai.dylib',
      'build/Release/libonnxruntime.dylib',
    ],
    win32: ['build/Release/onnxruntime-genai.dll', 'build/Release/onnxruntime.dll'],
  }[process.platform];
  assert.ok(expected, `unsupported packaging test platform: ${process.platform}`);
  for (const required of expected) {
    assert.ok(paths.includes(required), `npm package is missing ${required}`);
  }

  const extract = path.join(work, 'extracted');
  fs.mkdirSync(extract);
  run('tar', ['-xzf', path.join(work, packed.filename), '-C', extract]);
  const installedPackage = path.join(extract, 'package');
  const env = { ...process.env };
  delete env.LD_LIBRARY_PATH;
  delete env.DYLD_LIBRARY_PATH;
  const script = `
    const assert = require('node:assert/strict');
    const binding = require(${JSON.stringify(installedPackage)});
    assert.deepEqual(binding.__testRoundTrip({installed: true}), {installed: true});
  `;
  run(process.execPath, ['-e', script], { env, cwd: installedPackage });
});
