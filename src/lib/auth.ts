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
  // Regular user can only see their own data
  return { [userIdField]: userId };
}
