/** Browser/server shared conversion of a name to one safe path segment. */
export function toSafeSlug(input: string | null | undefined, fallback: string): string {
  const slug = (input ?? '')
    .replace(/\s+/g, '_')
    .replace(/[^a-zA-Z0-9_.-]/g, '')
    .replace(/^\.+/, '')
    .toLowerCase()
    .slice(0, 100)
  return slug.length > 0 ? slug : fallback
}
