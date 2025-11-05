# Getting Started - Admin Login Setup

## Step 1: Ensure Database is Set Up

First, make sure your database is initialized and migrations are run:

```powershell
cd frontend
npm run prisma:migrate:dev
```

This will:
- Create the database (if using SQLite, creates `prisma/dev.db`)
- Run all migrations to create the User table
- Generate the Prisma client

## Step 2: Create Your Admin Account

Use the values from your `.env.local` file (lines 7-9 should have `ADMIN_EMAIL` and `ADMIN_PASSWORD`):

```powershell
# From the frontend directory
$env:ADMIN_EMAIL="your-email@example.com"
$env:ADMIN_PASSWORD="your-password"
npm run prisma:seed
```

You should see:
```
✅ Admin user created/updated: { id: '...', email: '...', role: 'ADMIN' }
```

## Step 3: Verify Database Connection

Check that your `.env.local` has:
```env
DATABASE_URL="file:./prisma/dev.db"  # For SQLite (local dev)
# OR
DATABASE_URL="postgresql://postgres:postgres@localhost:5432/ragdb?schema=public"  # For PostgreSQL

SESSION_SECRET="your-secret-key-here-min-32-chars"
```

## Step 4: Start the Dev Server

```powershell
npm run dev
```

The app will be available at: http://localhost:3000

## Step 5: Login

1. Go to http://localhost:3000
2. You'll be redirected to `/login`
3. Enter the email and password you set in Step 2
4. You should be logged in and redirected to `/account`

## Troubleshooting

### "Invalid email or password"
- Make sure you ran the seed script with the correct credentials
- Check that the database exists and has the User table
- Verify your `.env.local` `DATABASE_URL` is correct

### "Database connection failed"
- For SQLite: Make sure `prisma/dev.db` exists
- For PostgreSQL: Make sure PostgreSQL is running and credentials are correct

### "Session not working"
- Check that `SESSION_SECRET` in `.env.local` is set (at least 32 characters)
- Clear browser cookies and try again

## Quick Check Commands

```powershell
# Check if database exists (SQLite)
Test-Path prisma/dev.db

# Check if migrations are applied
npm run prisma:studio
# Opens at http://localhost:5555 - you should see the User table
```

