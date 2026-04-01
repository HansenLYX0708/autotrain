# User Management System Setup

## Overview
A complete user management system has been added with the following features:

- **Authentication**: Login, register, logout, change password
- **Role-based access**: Admin and User roles
- **User Management**: Admin-only page for managing users (CRUD operations)
- **Data Isolation**: Regular users can only see their own projects, models, datasets, etc.
- **First-user admin**: The first registered user automatically becomes an admin

## Files Created/Modified

### Database Schema
- `prisma/schema.prisma` - Added User model and userId relations to all data models

### API Routes
- `src/app/api/auth/route.ts` - Authentication endpoints (login, register, logout, change password)
- `src/app/api/users/route.ts` - User management endpoints (admin only)
- `src/lib/auth.ts` - Authentication helpers for protecting APIs
- `src/app/api/projects/route.ts` - Updated with auth filtering

### Components & Contexts
- `src/contexts/auth-context.tsx` - React context for authentication state
- `src/components/pages/login.tsx` - Login/Register page
- `src/components/pages/users.tsx` - User management page (admin only)
- `src/components/change-password-dialog.tsx` - Change password dialog
- `src/app/page.tsx` - Updated to integrate auth system with user menu and admin navigation

## Setup Instructions

### Step 1: Run Database Migration

Since the database schema has been updated with new User model and relations, you need to run the migration:

```bash
# Reset the database (WARNING: This will delete all existing data)
bunx prisma migrate reset --force

# Or push the schema changes directly (may not work with SQLite drift)
bunx prisma db push --accept-data-loss

# Generate Prisma Client
bunx prisma generate
```

### Step 2: Start the Application

```bash
bun run dev
```

### Step 3: Create First Admin User

1. Open the application at `http://localhost:3000`
2. Click "Register" tab
3. Create the first user account - it will automatically become an **admin**
4. Subsequent registered users will be regular **users** by default

## User Roles

### Admin
- Can access User Management page
- Can create, edit, delete users
- Can see all projects, models, datasets from all users
- Can change any user's password

### User
- Cannot access User Management page
- Can only see their own projects, models, datasets, training configs
- Can change their own password

## Authentication Flow

1. **Unauthenticated users** see the Login/Register page
2. **After login**, users are redirected to the Dashboard
3. **Admin users** see an additional "User Management" menu item
4. **User menu** (top-right) shows:
   - Username and role
   - Change Password option
   - Logout option

## Security Notes

- Passwords are hashed using SHA-256 (in production, consider bcrypt with salt)
- Sessions are stored in memory (in production, use Redis)
- Cookie-based authentication with httpOnly flag
- API endpoints check authentication and authorization

## Future Enhancements

Consider adding:
1. Password reset via email
2. Session expiration warnings
3. Audit logs for user actions
4. Two-factor authentication
5. API rate limiting per user
