import { db } from '../src/lib/db'
import crypto from 'crypto'
async function main() {
  const password = crypto.createHash('sha256').update('e2e-pass-456').digest('hex')
  const u = await db.user.upsert({ where: { username: 'e2e_user2' },
    update: { role: 'user', status: 'active', password },
    create: { username: 'e2e_user2', password, role: 'user', status: 'active' } })
  console.log('e2e_user2', u.id)
}
main().finally(() => db['$disconnect']())
