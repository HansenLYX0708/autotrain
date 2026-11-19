/**
 * Browser APIs that exist only in secure contexts (https or localhost).
 * The platform is commonly served over plain http on a LAN IP, where
 * `crypto.randomUUID` and `navigator.clipboard` are undefined. Everything
 * below degrades gracefully instead of throwing.
 */

export function uuid(): string {
  const c = globalThis.crypto
  if (typeof c?.randomUUID === 'function') return c.randomUUID()
  const bytes = new Uint8Array(16)
  if (typeof c?.getRandomValues === 'function') {
    c.getRandomValues(bytes)
  } else {
    for (let i = 0; i < 16; i++) bytes[i] = Math.floor(Math.random() * 256)
  }
  bytes[6] = (bytes[6] & 0x0f) | 0x40 // version 4
  bytes[8] = (bytes[8] & 0x3f) | 0x80 // variant 10
  const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('')
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`
}

/** Copy text to the clipboard; never throws. Returns false when it could not copy. */
export async function copyText(text: string): Promise<boolean> {
  if (typeof window !== 'undefined' && window.isSecureContext) {
    try {
      await navigator.clipboard?.writeText(text)
      return true
    } catch {
      // fall through to the execCommand path
    }
  }
  try {
    const el = document.createElement('textarea')
    el.value = text
    el.style.position = 'fixed'
    el.style.top = '-1000px'
    el.style.opacity = '0'
    document.body.appendChild(el)
    el.focus()
    el.select()
    const ok = document.execCommand('copy')
    el.remove()
    return ok
  } catch {
    return false
  }
}
