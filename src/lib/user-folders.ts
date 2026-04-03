import { db } from "@/lib/db";
import * as fs from "fs";
import * as path from "path";

/**
 * Create user configuration folder structure
 * Creates: {userConfigsPath}/{username}/{datasets, models, training, jobs}
 */
export async function createUserConfigFolders(username: string): Promise<boolean> {
  try {
    // Get system config to find userConfigsPath
    const systemConfig = await db.systemConfig.findFirst();
    
    if (!systemConfig?.userConfigsPath) {
      console.warn(`[UserFolders] userConfigsPath not configured, skipping folder creation for ${username}`);
      return false;
    }

    const basePath = systemConfig.userConfigsPath;
    const userFolderPath = path.join(basePath, username);
    
    // Subfolders to create
    const subfolders = ["datasets", "models", "training", "jobs"];
    
    // Create user folder if not exists
    if (!fs.existsSync(userFolderPath)) {
      fs.mkdirSync(userFolderPath, { recursive: true });
      console.log(`[UserFolders] Created user folder: ${userFolderPath}`);
    }
    
    // Create subfolders
    for (const subfolder of subfolders) {
      const subfolderPath = path.join(userFolderPath, subfolder);
      if (!fs.existsSync(subfolderPath)) {
        fs.mkdirSync(subfolderPath, { recursive: true });
        console.log(`[UserFolders] Created subfolder: ${subfolderPath}`);
      }
    }
    
    return true;
  } catch (error) {
    console.error(`[UserFolders] Failed to create folders for user ${username}:`, error);
    return false;
  }
}

/**
 * Delete user configuration folder
 */
export async function deleteUserConfigFolders(username: string): Promise<boolean> {
  try {
    const systemConfig = await db.systemConfig.findFirst();
    
    if (!systemConfig?.userConfigsPath) {
      return false;
    }

    const userFolderPath = path.join(systemConfig.userConfigsPath, username);
    
    if (fs.existsSync(userFolderPath)) {
      fs.rmSync(userFolderPath, { recursive: true, force: true });
      console.log(`[UserFolders] Deleted user folder: ${userFolderPath}`);
    }
    
    return true;
  } catch (error) {
    console.error(`[UserFolders] Failed to delete folders for user ${username}:`, error);
    return false;
  }
}

/**
 * Create user database folder
 * Creates: {userDatabasePath}/{username}
 */
export async function createUserDatabaseFolder(username: string): Promise<boolean> {
  try {
    // Get system config to find userDatabasePath
    const systemConfig = await db.systemConfig.findFirst();
    
    if (!systemConfig?.userDatabasePath) {
      console.warn(`[UserFolders] userDatabasePath not configured, skipping database folder creation for ${username}`);
      return false;
    }

    const basePath = systemConfig.userDatabasePath;
    const userFolderPath = path.join(basePath, username);
    
    // Create user folder if not exists
    if (!fs.existsSync(userFolderPath)) {
      fs.mkdirSync(userFolderPath, { recursive: true });
      console.log(`[UserFolders] Created user database folder: ${userFolderPath}`);
    }
    
    return true;
  } catch (error) {
    console.error(`[UserFolders] Failed to create database folder for user ${username}:`, error);
    return false;
  }
}

/**
 * Delete user database folder
 */
export async function deleteUserDatabaseFolder(username: string): Promise<boolean> {
  try {
    const systemConfig = await db.systemConfig.findFirst();
    
    if (!systemConfig?.userDatabasePath) {
      return false;
    }

    const userFolderPath = path.join(systemConfig.userDatabasePath, username);
    
    if (fs.existsSync(userFolderPath)) {
      fs.rmSync(userFolderPath, { recursive: true, force: true });
      console.log(`[UserFolders] Deleted user database folder: ${userFolderPath}`);
    }
    
    return true;
  } catch (error) {
    console.error(`[UserFolders] Failed to delete database folder for user ${username}:`, error);
    return false;
  }
}
