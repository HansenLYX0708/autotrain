/**
 * Materialize built-in defaults and optional site templates for an offline update.
 * npx tsx scripts/export-default-configs.ts --output <dir> [--source <userConfigsPath>]
 */
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { isMap, parseDocument } from 'yaml';
import { FRAMEWORK_LIST } from '../src/lib/frameworks';
import { getBuiltinModelConfigs, getBuiltinTrainingConfigs, getDefaultFolderName } from '../src/lib/default-configs';

function argument(name: string) {
  const index = process.argv.indexOf(name);
  return index < 0 ? undefined : process.argv[index + 1];
}
const output = argument('--output');
const source = argument('--source');
if (!output) throw new Error('--output is required');
if (source && !fs.statSync(source).isDirectory()) throw new Error('Source config directory was not found');
const inventory: Array<{ framework: string; kind: string; path: string; origin: string; sha256: string }> = [];
for (const framework of FRAMEWORK_LIST) {
  const folder = getDefaultFolderName(framework);
  for (const kind of ['models', 'training'] as const) {
    const files = new Map<string, { content: string; origin: string }>();
    const builtins = kind === 'models' ? getBuiltinModelConfigs(framework) : getBuiltinTrainingConfigs(framework);
    for (const config of builtins) files.set(`${config.name}.yml`, { content: config.content, origin: 'built-in' });
    const local = source ? path.join(source, folder, kind) : '';
    if (local && fs.existsSync(local)) {
      for (const entry of fs.readdirSync(local, { withFileTypes: true })) {
        if (entry.isFile() && /\.(yaml|yml)$/i.test(entry.name)) {
          files.set(entry.name, { content: fs.readFileSync(path.join(local, entry.name), 'utf8'), origin: 'site-default' });
        }
      }
    }
    if (!files.size) throw new Error(`No ${kind} defaults for ${framework}`);
    for (const [name, file] of files) {
      const doc = parseDocument(file.content, { logLevel: 'silent' });
      if (doc.errors.length || !isMap(doc.contents)) throw new Error(`Invalid YAML: ${folder}/${kind}/${name}`);
      const relative = `${folder}/${kind}/${name}`;
      const target = path.join(output, relative);
      fs.mkdirSync(path.dirname(target), { recursive: true });
      fs.writeFileSync(target, file.content, 'utf8');
      inventory.push({ framework, kind, path: relative, origin: file.origin, sha256: createHash('sha256').update(file.content).digest('hex') });
    }
  }
}
fs.writeFileSync(path.join(output, 'inventory.json'), JSON.stringify(inventory, null, 2) + '\n', 'utf8');
console.log(`Exported ${inventory.length} default YAML files for ${FRAMEWORK_LIST.length} frameworks.`);
