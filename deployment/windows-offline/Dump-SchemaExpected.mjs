// Build-time helper: dump {table: [columns]} for every table of a database
// freshly pushed from the current Prisma schema.
// Usage: node Dump-SchemaExpected.mjs <dbPath> <appDir> <outFile>
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
const [database, app, outFile] = process.argv.slice(2);
if (!database || !app || !outFile || !fs.existsSync(database)) {
  throw new Error('Usage: Dump-SchemaExpected.mjs <dbPath> <appDir> <outFile>; database was not found');
}
const { PrismaClient } = createRequire(path.join(path.resolve(app), 'package.json'))('@prisma/client');
const client = new PrismaClient({ datasources: { db: { url: 'file:' + path.resolve(database).replaceAll('\\', '/') } } });
(async () => {
  try {
    const tables = await client.$queryRawUnsafe(
      "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name");
    const expected = {};
    for (const { name } of tables) {
      const info = await client.$queryRawUnsafe(`PRAGMA table_info("${name.replaceAll('"', '""')}")`);
      expected[name] = info.map(c => c.name);
    }
    fs.writeFileSync(outFile, JSON.stringify(expected, null, 1) + '\n');
    console.log(`Wrote ${Object.keys(expected).length} tables to ${outFile}`);
  } finally {
    await client.$disconnect();
  }
})().catch(error => { console.error(error.message); process.exitCode = 1; });
