import path from "path";

/**
 * Resolve a registered dataset's stored `datasetDir` to an absolute path.
 *
 * Upload-era datasets store a RELATIVE dir like `COCO/tiny`, conventionally
 * meaning `{userDatabasePath}/{ownerUsername}/{datasetDir}`. Admins can open
 * datasets owned by other users, so the owner — never the caller — supplies
 * the username segment. Returns null when the dir is relative and either
 * context value is missing (caller should fall back to its previous
 * behaviour).
 */
export function resolveDatasetRoot(
  datasetDir: string | null | undefined,
  userDatabasePath: string | null | undefined,
  ownerUsername: string | null | undefined,
): string | null {
  if (!datasetDir) return null;
  if (path.isAbsolute(datasetDir)) return datasetDir;
  if (!userDatabasePath || !ownerUsername) return null;
  return path.join(userDatabasePath, ownerUsername, datasetDir);
}
