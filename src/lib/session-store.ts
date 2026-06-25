/**
 * Process-wide session store.
 *
 * Next.js re-evaluates modules on hot-reload (dev) and can bundle route
 * segments separately, both of which reset or duplicate module-level state.
 * Storing the sessions Map on `globalThis` makes it a single shared instance
 * that survives HMR and is identical across every route bundle, so a token
 * created at login stays valid for all API routes (the same pattern used for
 * the Prisma client singleton).
 *
 * Note: this is still in-memory and is cleared on a full server restart; for
 * multi-instance/production deployments back it with Redis or a DB table.
 */

export interface SessionData {
  userId: string;
  role: string;
}

const globalForSessions = globalThis as unknown as {
  __authSessions?: Map<string, SessionData>;
};

export const sessions: Map<string, SessionData> =
  globalForSessions.__authSessions ?? new Map<string, SessionData>();

if (!globalForSessions.__authSessions) {
  globalForSessions.__authSessions = sessions;
}
