// Run semicolon-separated SQL statements from stdin against a sqlite database.
// Build/test helper only; used by New-UpdatePackage.ps1 smoke simulations.
// Usage: echo "SQL;" | node Exec-Sql.mjs <dbPath> <appDir>
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
const [database, app] = process.argv.slice(2);
if (!database || !app || !fs.existsSync(database)) throw new Error('Usage: Exec-Sql.mjs <dbPath> <appDir>');
const { PrismaClient } = createRequire(path.join(path.resolve(app), 'package.json'))('@prisma/client');
const client = new PrismaClient({ datasources: { db: { url: 'file:' + path.resolve(database).replaceAll('\\', '/') } } });
const sql = fs.readFileSync(0, 'utf8');
(async () => {
  try {
    for (const statement of sql.split(';').map(s => s.trim()).filter(Boolean)) {
      await client.$executeRawUnsafe(statement);
      console.log(`ok: ${statement.slice(0, 80)}`);
    }
  } finally {
    await client.$disconnect();
  }
})().catch(error => { console.error(error.message); process.exitCode = 1; });
