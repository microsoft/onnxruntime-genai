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
  const sourceManifest = JSON.parse(fs.readFileSync(path.join(packageRoot, 'package.json')));
  assert.equal(sourceManifest.name, 'onnxruntime-genai-non-generative');
  assert.equal(sourceManifest.os, undefined);
  assert.equal(sourceManifest.cpu, undefined);
  const paths = packed.files.map((file) => file.path);
  const addonPaths = paths.filter((file) =>
    /^build\/[^/]+\/onnxruntime_genai_node\.node$/.test(file),
  );
  assert.equal(addonPaths.length, 1);
  const nativeDirectory = path.posix.dirname(addonPaths[0]);

  const expected = {
    linux: ['libonnxruntime-genai.so', 'libonnxruntime.so'],
    darwin: [
      'libonnxruntime-genai.dylib',
      'libonnxruntime.dylib',
    ],
    win32: ['onnxruntime-genai.dll', 'onnxruntime.dll'],
  }[process.platform];
  assert.ok(expected, `unsupported packaging test platform: ${process.platform}`);
  for (const required of expected) {
    const packagedPath = path.posix.join(nativeDirectory, required);
    assert.ok(paths.includes(packagedPath), `npm package is missing ${packagedPath}`);
  }

  const extract = path.join(work, 'extracted');
  fs.mkdirSync(extract);
  run('tar', ['-xzf', path.join(work, packed.filename), '-C', extract]);
  const installedPackage = path.join(extract, 'package');
  const manifest = JSON.parse(fs.readFileSync(path.join(installedPackage, 'package.json')));
  assert.equal(
    manifest.name,
    `onnxruntime-genai-non-generative-${process.platform}-${process.arch}`,
  );
  assert.deepEqual(manifest.os, [process.platform]);
  assert.deepEqual(manifest.cpu, [process.arch]);
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
