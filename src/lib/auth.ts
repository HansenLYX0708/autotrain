import { NextRequest, NextResponse } from "next/server";
import { sessions } from "@/app/api/auth/route";
import { db } from "@/lib/db";

export interface AuthContext {
  userId: string;
  role: 'admin' | 'user';
  isAuthenticated: boolean;
}

// Helper to get current user from request
export async function getCurrentUser(request: NextRequest): Promise<{ userId: string; role: string } | null> {
  const token = request.cookies.get("auth-token")?.value;
  
  if (!token || !sessions.has(token)) {
    return null;
  }

  const session = sessions.get(token)!;
  
  // Verify user still exists and is active
  const user = await db.user.findUnique({
    where: { id: session.userId },
    select: { id: true, role: true, status: true },
  });

  if (!user || user.status !== "active") {
    sessions.delete(token);
    return null;
  }

  return { userId: user.id, role: user.role };
}

// Helper to require authentication
export async function requireAuth(request: NextRequest): Promise<{ userId: string; role: string } | NextResponse> {
  const user = await getCurrentUser(request);
  
  if (!user) {
    return NextResponse.json({ error: "Unauthorized - Please login" }, { status: 401 });
  }

  return user;
}

// Helper to require admin role
export async function requireAdmin(request: NextRequest): Promise<{ userId: string; role: string } | NextResponse> {
  const user = await getCurrentUser(request);
  
  if (!user) {
    return NextResponse.json({ error: "Unauthorized - Please login" }, { status: 401 });
  }

  if (user.role !== "admin") {
    return NextResponse.json({ error: "Forbidden - Admin access required" }, { status: 403 });
  }

  return user;
}

// Helper to build where clause for user data filtering
export function buildUserFilter(userId: string, role: string, userIdField: string = 'userId') {
  if (role === 'admin') {
    return {}; // Admin can see all data
  }
  // Regular user can see their own data OR data with no owner (userId is null)
  return { 
    OR: [
      { [userIdField]: userId },
      { [userIdField]: null }
    ]
  };
}

/**
 * Authenticate the request and build a Prisma `where` clause that scopes a
 * single record lookup to what the caller is allowed to touch.
 *
 * Use this in every `[id]` route. Fetching by bare `{ id }` and then acting on
 * the result is an IDOR: the id is guessable/enumerable and the row may belong
 * to another user. Admins get unrestricted access; everyone else is limited to
 * rows they own, plus legacy rows with no owner (matching `buildUserFilter`).
 *
 * Returns a 401 `NextResponse` when unauthenticated, so callers should do:
 *
 *     const scope = await requireOwnedScope(request, id);
 *     if (scope instanceof NextResponse) return scope;
 *     const row = await db.thing.findFirst({ where: scope.where });
 *     if (!row) return notFound();
 */
export async function requireOwnedScope(
  request: NextRequest,
  id: string,
  userIdField: string = 'userId'
): Promise<{ userId: string; role: string; where: Record<string, unknown> } | NextResponse> {
  const auth = await requireAuth(request);
  if (auth instanceof NextResponse) return auth;

  const { userId, role } = auth;
  const where: Record<string, unknown> =
    role === 'admin'
      ? { id }
      : { id, OR: [{ [userIdField]: userId }, { [userIdField]: null }] };

  return { userId, role, where };
}

/** Standard 404 for "does not exist, or you may not touch it". */
export function notFoundOrDenied(entity: string): NextResponse {
  return NextResponse.json(
    { success: false, error: `${entity} not found or access denied` },
    { status: 404 }
  );
}
