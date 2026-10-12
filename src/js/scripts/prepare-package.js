'use strict';

const fs = require('node:fs');
const path = require('node:path');

const packagePath = path.resolve(__dirname, '..', 'package.json');
const backupPath = path.resolve(__dirname, '..', '.package.json.platform-backup');

if (process.argv[2] === 'prepare') {
  if (!['linux', 'darwin', 'win32'].includes(process.platform)) {
    throw new Error(`Unsupported npm package platform: ${process.platform}`);
  }
  if (!['x64', 'arm64'].includes(process.arch)) {
    throw new Error(`Unsupported npm package architecture: ${process.arch}`);
  }
  if (fs.existsSync(backupPath)) {
    fs.copyFileSync(backupPath, packagePath);
  }
  fs.copyFileSync(packagePath, backupPath);
  const manifest = JSON.parse(fs.readFileSync(packagePath, 'utf8'));
  manifest.name = `${manifest.name}-${process.platform}-${process.arch}`;
  manifest.os = [process.platform];
  manifest.cpu = [process.arch];
  fs.writeFileSync(packagePath, `${JSON.stringify(manifest, null, 2)}\n`);
} else if (process.argv[2] === 'restore') {
  if (fs.existsSync(backupPath)) {
    fs.copyFileSync(backupPath, packagePath);
    fs.rmSync(backupPath);
  }
} else {
  throw new Error('Expected prepare or restore');
}
