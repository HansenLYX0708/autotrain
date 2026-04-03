import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import * as crypto from "crypto";
import { createUserConfigFolders, createUserDatabaseFolder } from "@/lib/user-folders";

// Simple in-memory session store (in production, use Redis or similar)
const sessions = new Map<string, { userId: string; role: string }>();

// Helper to hash password
function hashPassword(password: string): string {
  return crypto.createHash("sha256").update(password).digest("hex");
}

// Helper to generate session token
function generateToken(): string {
  return crypto.randomBytes(32).toString("hex");
}

// POST /api/auth/login
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { action } = body;

    switch (action) {
      case "login":
        return handleLogin(body);
      case "register":
        return handleRegister(body);
      case "logout":
        return handleLogout(request);
      case "changePassword":
        return handleChangePassword(request, body);
      default:
        return NextResponse.json({ error: "Invalid action" }, { status: 400 });
    }
  } catch (error) {
    console.error("Auth error:", error);
    return NextResponse.json(
      { error: "Authentication failed", message: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}

// GET /api/auth/me - Get current user
export async function GET(request: NextRequest) {
  try {
    const token = request.cookies.get("auth-token")?.value;
    
    if (!token || !sessions.has(token)) {
      return NextResponse.json({ error: "Not authenticated" }, { status: 401 });
    }

    const session = sessions.get(token)!;
    const user = await db.user.findUnique({
      where: { id: session.userId },
      select: { id: true, username: true, role: true, status: true, createdAt: true },
    });

    if (!user || user.status !== "active") {
      sessions.delete(token);
      return NextResponse.json({ error: "User not found or disabled" }, { status: 401 });
    }

    return NextResponse.json({ success: true, data: user });
  } catch (error) {
    console.error("Get current user error:", error);
    return NextResponse.json(
      { error: "Failed to get user info", message: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}

async function handleLogin({ username, password }: { username: string; password: string }) {
  if (!username || !password) {
    return NextResponse.json({ error: "Username and password are required" }, { status: 400 });
  }

  const user = await db.user.findUnique({ where: { username } });

  if (!user || user.password !== hashPassword(password)) {
    return NextResponse.json({ error: "Invalid username or password" }, { status: 401 });
  }

  if (user.status !== "active") {
    return NextResponse.json({ error: "Account is disabled" }, { status: 403 });
  }

  // Create session
  const token = generateToken();
  sessions.set(token, { userId: user.id, role: user.role });

  // Set cookie
  const response = NextResponse.json({
    success: true,
    data: {
      id: user.id,
      username: user.username,
      role: user.role,
    },
  });

  response.cookies.set("auth-token", token, {
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "strict",
    maxAge: 60 * 60 * 24 * 7, // 7 days
    path: "/",
  });

  return response;
}

async function handleRegister({ username, password }: { username: string; password: string }) {
  if (!username || !password) {
    return NextResponse.json({ error: "Username and password are required" }, { status: 400 });
  }

  if (password.length < 6) {
    return NextResponse.json({ error: "Password must be at least 6 characters" }, { status: 400 });
  }

  // Check if username already exists
  const existing = await db.user.findUnique({ where: { username } });
  if (existing) {
    return NextResponse.json({ error: "Username already exists" }, { status: 400 });
  }

  // Check if this is the first user (make them admin)
  const userCount = await db.user.count();
  const role = userCount === 0 ? "admin" : "user";

  const user = await db.user.create({
    data: {
      username,
      password: hashPassword(password),
      role,
      status: "active",
    },
  });

  // Create user config folders and database folder
  await createUserConfigFolders(username);
  await createUserDatabaseFolder(username);

  return NextResponse.json({
    success: true,
    data: {
      id: user.id,
      username: user.username,
      role: user.role,
    },
  });
}

async function handleLogout(request: NextRequest) {
  const token = request.cookies.get("auth-token")?.value;
  
  if (token) {
    sessions.delete(token);
  }

  const response = NextResponse.json({ success: true, message: "Logged out successfully" });
  response.cookies.set("auth-token", "", { maxAge: 0, path: "/" });
  
  return response;
}

async function handleChangePassword(
  request: NextRequest,
  { oldPassword, newPassword }: { oldPassword: string; newPassword: string }
) {
  const token = request.cookies.get("auth-token")?.value;
  
  if (!token || !sessions.has(token)) {
    return NextResponse.json({ error: "Not authenticated" }, { status: 401 });
  }

  if (!oldPassword || !newPassword) {
    return NextResponse.json({ error: "Old password and new password are required" }, { status: 400 });
  }

  if (newPassword.length < 6) {
    return NextResponse.json({ error: "New password must be at least 6 characters" }, { status: 400 });
  }

  const session = sessions.get(token)!;
  const user = await db.user.findUnique({ where: { id: session.userId } });

  if (!user || user.password !== hashPassword(oldPassword)) {
    return NextResponse.json({ error: "Old password is incorrect" }, { status: 400 });
  }

  await db.user.update({
    where: { id: user.id },
    data: { password: hashPassword(newPassword) },
  });

  return NextResponse.json({ success: true, message: "Password changed successfully" });
}

// Export sessions map for use in other routes
export { sessions };
