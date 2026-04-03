import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import * as fs from "fs";
import * as path from "path";
import { sessions } from "../../auth/route";

// Helper to get folder size recursively
function getFolderSize(folderPath: string): number {
  let totalSize = 0;
  
  if (!fs.existsSync(folderPath)) {
    return 0;
  }
  
  const stats = fs.statSync(folderPath);
  
  if (stats.isFile()) {
    return stats.size;
  }
  
  const files = fs.readdirSync(folderPath);
  for (const file of files) {
    const filePath = path.join(folderPath, file);
    const fileStats = fs.statSync(filePath);
    
    if (fileStats.isDirectory()) {
      totalSize += getFolderSize(filePath);
    } else {
      totalSize += fileStats.size;
    }
  }
  
  return totalSize;
}

// GET /api/users/storage - Get current user's storage info
export async function GET(request: NextRequest) {
  try {
    // Get current user from session
    const token = request.cookies.get("auth-token")?.value;
    if (!token || !sessions.has(token)) {
      return NextResponse.json({ error: "Not authenticated" }, { status: 401 });
    }
    
    const session = sessions.get(token)!;
    const userId = session.userId;
    
    // Get user info
    const user = await db.user.findUnique({
      where: { id: userId },
      select: { id: true, username: true, maxStorageQuota: true, role: true },
    });
    
    if (!user) {
      return NextResponse.json({ error: "User not found" }, { status: 404 });
    }
    
    // Get system config for userDatabasePath
    const systemConfig = await db.systemConfig.findFirst();
    const userDatabasePath = systemConfig?.userDatabasePath;
    
    let usedStorage = 0;
    
    if (userDatabasePath) {
      const userFolderPath = path.join(userDatabasePath, user.username);
      usedStorage = getFolderSize(userFolderPath);
    }
    
    return NextResponse.json({
      success: true,
      data: {
        userId: user.id,
        username: user.username,
        maxStorageQuota: user.maxStorageQuota.toString(), // BigInt needs to be converted to string for JSON
        usedStorage,
        availableStorage: Number(user.maxStorageQuota) - usedStorage,
        userDatabasePath: userDatabasePath ? path.join(userDatabasePath, user.username) : null,
      },
    });
  } catch (error) {
    console.error("Error getting storage info:", error);
    return NextResponse.json(
      { error: "Failed to get storage info", message: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}

// Admin endpoint to get any user's storage info
// GET /api/users/storage?userId=xxx
export async function POST(request: NextRequest) {
  try {
    // Check if admin
    const token = request.cookies.get("auth-token")?.value;
    if (!token || !sessions.has(token)) {
      return NextResponse.json({ error: "Not authenticated" }, { status: 401 });
    }
    
    const session = sessions.get(token)!;
    if (session.role !== "admin") {
      return NextResponse.json({ error: "Unauthorized - Admin only" }, { status: 403 });
    }
    
    const body = await request.json();
    const { userId, username } = body;
    
    let targetUser;
    
    if (userId) {
      targetUser = await db.user.findUnique({
        where: { id: userId },
        select: { id: true, username: true, maxStorageQuota: true },
      });
    } else if (username) {
      targetUser = await db.user.findUnique({
        where: { username },
        select: { id: true, username: true, maxStorageQuota: true },
      });
    } else {
      return NextResponse.json({ error: "userId or username is required" }, { status: 400 });
    }
    
    if (!targetUser) {
      return NextResponse.json({ error: "User not found" }, { status: 404 });
    }
    
    // Get system config for userDatabasePath
    const systemConfig = await db.systemConfig.findFirst();
    const userDatabasePath = systemConfig?.userDatabasePath;
    
    let usedStorage = 0;
    
    if (userDatabasePath) {
      const userFolderPath = path.join(userDatabasePath, targetUser.username);
      usedStorage = getFolderSize(userFolderPath);
    }
    
    return NextResponse.json({
      success: true,
      data: {
        userId: targetUser.id,
        username: targetUser.username,
        maxStorageQuota: targetUser.maxStorageQuota.toString(),
        usedStorage,
        availableStorage: Number(targetUser.maxStorageQuota) - usedStorage,
        userDatabasePath: userDatabasePath ? path.join(userDatabasePath, targetUser.username) : null,
      },
    });
  } catch (error) {
    console.error("Error getting user storage info:", error);
    return NextResponse.json(
      { error: "Failed to get user storage info", message: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}
