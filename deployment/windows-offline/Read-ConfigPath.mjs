// Read the installed database without modifying settings or creating a new DB.
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
const [database, app] = process.argv.slice(2);
if (!database || !app || !fs.existsSync(database)) throw new Error('Installed database was not found');
const { PrismaClient } = createRequire(path.join(path.resolve(app), 'package.json'))('@prisma/client');
const client = new PrismaClient({ datasources: { db: { url: 'file:' + path.resolve(database).replaceAll('\\', '/') } } });
(async () => {
  try {
    const rows = await client.$queryRawUnsafe('SELECT "userConfigsPath" FROM "SystemConfig" ORDER BY "createdAt" LIMIT 1');
    if (rows[0]?.userConfigsPath) process.stdout.write(rows[0].userConfigsPath);
  } finally {
    await client.$disconnect();
  }
})().catch(error => { console.error(error.message); process.exitCode = 1; });
