import { PrismaClient } from '@prisma/client'
import * as fs from 'fs'
import * as path from 'path'

/**
 * Fail loudly when DATABASE_URL points at a SQLite file that does not exist.
 *
 * SQLite happily creates a new, empty database file on first connect. Combined
 * with a misconfigured DATABASE_URL that means the app silently comes up with
 * zero users and zero projects instead of reporting a problem — which is
 * exactly what happened here: `.env` shipped a leftover Linux path
 * (`file:/home/z/my-project/db/custom.db`) from the original scaffold. On
 * Windows that resolves against the *current drive*, so the real data lived at
 * `D:\home\z\...` and launching from another drive would have quietly used a
 * different, empty database.
 *
 * We warn rather than throw so that a genuine first-time setup (`prisma db
 * push` against a fresh file) still works.
 */
function warnIfDatabaseMissing(): void {
  const url = process.env.DATABASE_URL
  if (!url) {
    console.error('[db] DATABASE_URL is not set. Prisma will fail to connect.')
    return
  }
  if (!url.startsWith('file:')) return

  const raw = url.slice('file:'.length)
  // Prisma resolves relative SQLite paths against the schema directory.
  const resolved = path.isAbsolute(raw) ? raw : path.resolve(process.cwd(), 'prisma', raw)

  if (!fs.existsSync(resolved)) {
    console.error(
      '\n' +
        '='.repeat(78) + '\n' +
        '[db] WARNING: the SQLite database does not exist yet.\n' +
        `      DATABASE_URL : ${url}\n` +
        `      resolves to  : ${resolved}\n` +
        '      SQLite will create an EMPTY database here, so the app will start with\n' +
        '      no users and no projects. If you expected existing data, DATABASE_URL is\n' +
        '      pointing at the wrong place — check it before logging in.\n' +
        '      Note: a relative "file:..." path is resolved against the prisma/ folder,\n' +
        '      and a path like "file:/foo/bar" is resolved against the current DRIVE on\n' +
        '      Windows. Prefer a fully-qualified path, e.g. file:D:/path/to/custom.db\n' +
        '='.repeat(78) + '\n',
    )
  }
}

const globalForPrisma = globalThis as unknown as {
  prisma: PrismaClient | undefined
  prismaChecked: boolean | undefined
}

if (!globalForPrisma.prismaChecked) {
  globalForPrisma.prismaChecked = true
  warnIfDatabaseMissing()
}

export const db =
  globalForPrisma.prisma ??
  new PrismaClient({
    // `query` logging prints every SQL statement. Useful while debugging, very
    // noisy otherwise, so keep it out of production logs.
    log: process.env.NODE_ENV === 'production' ? ['warn', 'error'] : ['query', 'warn', 'error'],
  })

if (process.env.NODE_ENV !== 'production') globalForPrisma.prisma = db
