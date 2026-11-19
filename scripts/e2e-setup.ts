import { db } from '../src/lib/db'
import crypto from 'crypto'

async function main() {
  const sys = await db.systemConfig.findFirst()
  console.log('SystemConfig:', JSON.stringify({
    userDatabasePath: (sys as any)?.userDatabasePath,
    torchPath: (sys as any)?.torchPath,
    frameworkPythonMappings: (sys as any)?.frameworkPythonMappings,
  }))
  const password = crypto.createHash('sha256').update('e2e-pass-123').digest('hex')
  const user = await db.user.upsert({
    where: { username: 'e2e_admin' },
    update: { role: 'admin', status: 'active', password, maxStorageQuota: BigInt(200 * 1024 * 1024 * 1024) },
    create: { username: 'e2e_admin', password, role: 'admin', status: 'active', maxStorageQuota: BigInt(200 * 1024 * 1024 * 1024) },
  })
  console.log('e2e_admin id:', user.id)
}
main().finally(() => db['$disconnect']())
