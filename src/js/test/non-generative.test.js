'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');
const path = require('node:path');
const {
  DirectoryTokenizer,
  RankingSession,
  DecisionSession,
  __testRoundTrip,
} = require('..');

test('structured values use native builders and accessors', () => {
  const value = {
    first: 1,
    second: [null, true, 1.25, 9007199254740992n],
    third: { nested: 'value' },
  };
  const result = __testRoundTrip(value);
  assert.deepEqual(Object.keys(result), ['first', 'second', 'third']);
  assert.deepEqual(result, value);
});

test('structured numeric validation is lossless', () => {
  assert.throws(() => __testRoundTrip(Number.MAX_SAFE_INTEGER + 1), /safe integer/);
  assert.throws(() => __testRoundTrip(2n ** 63n), /signed 64-bit/);
  assert.throws(() => __testRoundTrip(NaN), /finite/);
  assert.throws(() => __testRoundTrip(Symbol('bad')), /Unsupported structured/);
});

test('structured cycles and excessive depth fail without crashing', () => {
  const packageRoot = path.resolve(__dirname, '..');
  const script = `
    const assert = require('node:assert/strict');
    const { __testRoundTrip } = require(${JSON.stringify(packageRoot)});
    const objectCycle = {};
    objectCycle.self = objectCycle;
    assert.throws(() => __testRoundTrip(objectCycle), {
      name: 'TypeError',
      message: /cycles/,
    });
    const arrayCycle = [];
    arrayCycle.push(arrayCycle);
    assert.throws(() => __testRoundTrip(arrayCycle), {
      name: 'TypeError',
      message: /cycles/,
    });
    let deep = null;
    for (let i = 0; i < 129; ++i) deep = [deep];
    assert.throws(() => __testRoundTrip(deep), {
      name: 'TypeError',
      message: /maximum nesting depth/,
    });
  `;
  const child = spawnSync(process.execPath, ['-e', script], {
    encoding: 'utf8',
    env: process.env,
  });
  assert.equal(child.status, 0, `child crashed or failed:\n${child.stderr}`);

  const shared = { value: 7 };
  assert.deepEqual(__testRoundTrip({ first: shared, second: shared }), {
    first: { value: 7 },
    second: { value: 7 },
  });
});

test('dangerous structured keys remain own data properties', () => {
  const input = JSON.parse(
    '{"__proto__":{"polluted":true},"constructor":"safe","prototype":"also-safe"}',
  );
  const result = __testRoundTrip(input);
  assert.equal(Object.getPrototypeOf(result), Object.prototype);
  for (const key of ['__proto__', 'constructor', 'prototype']) {
    const descriptor = Object.getOwnPropertyDescriptor(result, key);
    assert.ok(descriptor, `${key} must be an own property`);
    assert.equal(descriptor.enumerable, true);
    assert.equal(descriptor.writable, true);
    assert.equal(descriptor.configurable, true);
  }
  assert.deepEqual(result.__proto__, { polluted: true });
  assert.equal(result.constructor, 'safe');
  assert.equal(result.prototype, 'also-safe');
  assert.equal({}.polluted, undefined);
});

test('structured conversion ignores inherited enumerable properties', () => {
  const prototype = { inherited: 'exclude-me' };
  const input = Object.create(prototype);
  input.own = 'include-me';
  assert.deepEqual(__testRoundTrip(input), { own: 'include-me' });
});

test('constructors reject invalid input', () => {
  assert.throws(() => new DirectoryTokenizer(), /string/);
  assert.throws(() => new RankingSession('', ['cpu', 3]), /provider/i);
  assert.throws(() => new DecisionSession('', 'cpu'), /array/i);
});

test(
  'opt-in CLM lifecycle, cache and free-form ranking',
  { skip: !process.env.ORTGENAI_TEST_CLM_PACKAGE },
  () => {
  const packagePath = process.env.ORTGENAI_TEST_CLM_PACKAGE;
  const session = new RankingSession(packagePath, ['cpu']);
  const request = {
    state: { weather: 'rain' },
    questions: { q: { type: 'noul', instructions: 'Is this suitable?' } },
  };
  session.setCacheCapacity(2, 1024 * 1024);
  assert.equal(session.run(request).answers.length, 1);
  assert.equal(
    session.rank({
      state: request.state,
      instructions: 'Choose one',
      candidates: { inside: 'museum', outside: 'picnic' },
    }).items.length,
    2,
  );
  assert.equal(session.cacheStats.entryCapacity, 2);
  session.clearCache();
  assert.equal(session.cacheStats.entries, 0);
  session.invalidateCache();

  const reentrant = new RankingSession(packagePath, ['cpu']);
  const reentrantRequest = {
    questions: { q: { type: 'noul', instructions: 'Is this suitable?' } },
  };
  Object.defineProperty(reentrantRequest, 'state', {
    enumerable: true,
    get() {
      reentrant.close();
      return { weather: 'rain' };
    },
  });
  assert.throws(() => reentrant.run(reentrantRequest), /closed/);

  session.close();
  session.close();
  assert.throws(() => session.run(request), /closed/);
  },
);

test(
  'opt-in KEV lifecycle, prefix reuse and tokenizer',
  { skip: !process.env.ORTGENAI_TEST_KEV_PACKAGE },
  () => {
  const packagePath = process.env.ORTGENAI_TEST_KEV_PACKAGE;
  const tokenizer = new DirectoryTokenizer(packagePath);
  assert.ok(tokenizer.encode('rain') instanceof Int32Array);
  const session = new DecisionSession(packagePath, ['cpu']);
  const request = {
    state: 'rain',
    questions: { q: { type: 'noul', instructions: 'Take an umbrella?' } },
  };
  assert.equal(session.decide(request).answers.length, 1);
  session.prefixReuseEnabled = false;
  assert.equal(session.prefixReuseEnabled, false);
  assert.equal(typeof session.prefixReuseStatus, 'string');
  session.setPrefixCacheCapacity(2, 1024 * 1024);
  assert.equal(session.prefixCacheStats.entryCapacity, 2);
  assert.equal(typeof session.prefixReuseStats.prefixRuns, 'bigint');
  session.close();
  tokenizer.close();
  assert.throws(() => tokenizer.encode('rain'), /closed/);
  },
);
