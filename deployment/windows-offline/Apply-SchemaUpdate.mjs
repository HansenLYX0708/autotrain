// Apply additive, idempotent schema statements to the installed database, then
// verify the DB satisfies schema-expected.json (table -> column names).
// Usage: node Apply-SchemaUpdate.mjs <dbPath> <appDir> <packageRoot>
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
const [database, app, packageRoot] = process.argv.slice(2);
if (!database || !app || !packageRoot || !fs.existsSync(database)) {
  throw new Error('Usage: Apply-SchemaUpdate.mjs <dbPath> <appDir> <packageRoot>; installed database was not found');
}
const { PrismaClient } = createRequire(path.join(path.resolve(app), 'package.json'))('@prisma/client');
const client = new PrismaClient({ datasources: { db: { url: 'file:' + path.resolve(database).replaceAll('\\', '/') } } });
const additions = JSON.parse(fs.readFileSync(path.join(packageRoot, 'schema-additions.json'), 'utf8'));
const expected = JSON.parse(fs.readFileSync(path.join(packageRoot, 'schema-expected.json'), 'utf8'));
(async () => {
  try {
    for (const { table, statements } of additions) {
      const existing = await client.$queryRawUnsafe(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = $1", table);
      if (existing.length) {
        console.log(`Table ${table}: already present, statements skipped`);
        continue;
      }
      for (const statement of statements) {
        await client.$executeRawUnsafe(statement);
      }
      console.log(`Table ${table}: created (${statements.length} statements)`);
    }
    const missing = [];
    for (const [table, columns] of Object.entries(expected)) {
      const info = await client.$queryRawUnsafe(`PRAGMA table_info("${table.replaceAll('"', '""')}")`);
      if (!info.length) { missing.push(`table ${table}`); continue; }
      const present = new Set(info.map(c => c.name));
      for (const column of columns) {
        if (!present.has(column)) missing.push(`${table}.${column}`);
      }
    }
    if (missing.length) {
      console.error('Schema verification failed; missing:');
      for (const item of missing) console.error(`  - ${item}`);
      process.exitCode = 1;
      return;
    }
    console.log(`Schema verification passed (${Object.keys(expected).length} tables).`);
  } finally {
    await client.$disconnect();
  }
})().catch(error => { console.error(error.message); process.exitCode = 1; });
