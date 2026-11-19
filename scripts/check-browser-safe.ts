import { uuid } from '../src/lib/browser-safe'

const V4 = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/
let failures = 0
function check(name: string, ok: boolean) {
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${name}`)
  if (!ok) failures++
}

console.log('[1] uuid() native path')
check('v4 format', V4.test(uuid()))
const set = new Set(Array.from({ length: 1000 }, () => uuid()))
check('1000 unique', set.size === 1000)

const cryptoObj = globalThis.crypto as Crypto & { randomUUID?: () => string }
const originalRandomUUID = cryptoObj?.randomUUID
const originalGetRandomValues = cryptoObj?.getRandomValues

console.log('[2] uuid() getRandomValues fallback (randomUUID removed)')
try {
  Object.defineProperty(cryptoObj, 'randomUUID', { value: undefined, configurable: true, writable: true })
  const u = uuid()
  check('v4 format via getRandomValues', V4.test(u))
  const s2 = new Set(Array.from({ length: 1000 }, () => uuid()))
  check('1000 unique', s2.size === 1000)
} finally {
  Object.defineProperty(cryptoObj, 'randomUUID', { value: originalRandomUUID, configurable: true, writable: true })
}

console.log('[3] uuid() Math.random fallback (crypto methods removed)')
try {
  Object.defineProperty(cryptoObj, 'randomUUID', { value: undefined, configurable: true, writable: true })
  Object.defineProperty(cryptoObj, 'getRandomValues', { value: undefined, configurable: true, writable: true })
  const u = uuid()
  check('v4 format via Math.random', V4.test(u))
  const s3 = new Set(Array.from({ length: 1000 }, () => uuid()))
  check('1000 unique', s3.size === 1000)
} finally {
  Object.defineProperty(cryptoObj, 'randomUUID', { value: originalRandomUUID, configurable: true, writable: true })
  Object.defineProperty(cryptoObj, 'getRandomValues', { value: originalGetRandomValues, configurable: true, writable: true })
}

console.log(failures === 0 ? 'ALL PASS' : `${failures} FAILED`)
process.exit(failures === 0 ? 0 : 1)
