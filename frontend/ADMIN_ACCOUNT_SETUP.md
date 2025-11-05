# Admin Account Setup Guide

This guide explains how to create an admin account using the command line seed script.

## Prerequisites

1. All dependencies installed: `npm install`
2. Prisma client generated: `npx prisma generate`
3. Database migrated: `npx prisma migrate dev`
4. Environment variables set in `.env` file

## Creating an Admin Account

### Step 1: Set Environment Variables

Create a `.env` file in the `frontend` directory (or copy from `.env.example`) and set:

```bash
ADMIN_EMAIL="admin@example.com"
ADMIN_PASSWORD="your-secure-password"
```

**Important**: Choose a strong password for your admin account in production!

### Step 2: Run the Seed Script

Execute the seed script with the environment variables:

**Windows (PowerShell):**
```powershell
$env:ADMIN_EMAIL="admin@example.com"; $env:ADMIN_PASSWORD="admin123"; npm run prisma:seed
```

**Windows (CMD):**
```cmd
set ADMIN_EMAIL=admin@example.com && set ADMIN_PASSWORD=admin123 && npm run prisma:seed
```

**Linux/Mac:**
```bash
ADMIN_EMAIL="admin@example.com" ADMIN_PASSWORD="admin123" npm run prisma:seed
```

### Step 3: Verify Admin Account

The seed script will:
- Create a new admin user if one doesn't exist
- Update the password if the admin user already exists (upsert behavior)

You should see output like:
```
✅ Admin user created/updated: {
  id: 'clx...',
  email: 'admin@example.com',
  role: 'ADMIN'
}
```

### Step 4: Login

1. Start your dev server: `npm run dev`
2. Navigate to: `http://localhost:3000/login`
3. Login with your admin credentials:
   - Email: `admin@example.com` (or whatever you set)
   - Password: `admin123` (or whatever you set)

## Updating Admin Password

To update the admin password, simply run the seed script again with a new password:

```bash
ADMIN_EMAIL="admin@example.com" ADMIN_PASSWORD="new-password" npm run prisma:seed
```

The script uses Prisma's `upsert` operation, so it will update the existing admin user's password.

## Troubleshooting

### Error: "ADMIN_EMAIL and ADMIN_PASSWORD environment variables are required"

**Solution**: Make sure you're setting the environment variables before running the seed script. The format depends on your shell (see examples above).

### Error: "Module not found: @prisma/client"

**Solution**: Run `npx prisma generate` first to generate the Prisma client.

### Error: "Cannot find module 'prisma/seed.ts'"

**Solution**: Make sure you're running the command from the `frontend` directory where `prisma/seed.ts` exists.

### Database not found

**Solution**: Run `npx prisma migrate dev` first to create the database and schema.

## Complete Setup Commands

Here's the complete sequence of commands to set up everything from scratch:

```bash
# 1. Install dependencies
npm install

# 2. Generate Prisma client
npx prisma generate

# 3. Create database and run migrations
npx prisma migrate dev --name init

# 4. Create admin account (set your own email/password)
ADMIN_EMAIL="admin@example.com" ADMIN_PASSWORD="admin123" npm run prisma:seed

# 5. Start dev server
npm run dev
```

## Security Notes

- **Never commit your `.env` file** to version control
- Use strong passwords for admin accounts in production
- Change the default `SESSION_SECRET` in production
- Consider using environment-specific `.env` files (`.env.local`, `.env.production`, etc.)

## Production Deployment

For production, set environment variables in your hosting platform (Vercel, Railway, etc.) rather than using a `.env` file. The seed script will work the same way, reading from environment variables.

