# Admin User Creation Guide

## Registration Disabled

Public registration is **disabled**. Only administrators can create user accounts.

## How to Create User Accounts

### Option 1: Using Prisma Seed Script (Recommended)

Create a custom seed script or modify the existing one to add users:

```powershell
# Navigate to frontend directory
cd frontend

# Create a new seed script or modify prisma/seed.ts
# Then run:
npm run prisma:seed
```

### Option 2: Using Prisma Studio (GUI)

1. Run Prisma Studio:
   ```powershell
   cd frontend
   npm run prisma:studio
   ```

2. Open http://localhost:5555 in your browser
3. Click on "User" model
4. Click "Add record"
5. Fill in:
   - `email`: User's email address
   - `passwordHash`: Generate using Node.js:
     ```javascript
     const bcrypt = require('bcrypt');
     bcrypt.hash('user_password', 10).then(hash => console.log(hash));
     ```
   - `role`: Either `USER` or `ADMIN`
6. Click "Save 1 change"

### Option 3: Using Node.js Script

Create a script `create-user.js`:

```javascript
const { PrismaClient } = require('@prisma/client');
const bcrypt = require('bcrypt');

const prisma = new PrismaClient();

async function createUser() {
  const email = process.argv[2]; // Get from command line
  const password = process.argv[3]; // Get from command line
  const role = process.argv[4] || 'USER'; // Optional: USER or ADMIN

  if (!email || !password) {
    console.error('Usage: node create-user.js <email> <password> [role]');
    process.exit(1);
  }

  const passwordHash = await bcrypt.hash(password, 10);

  const user = await prisma.user.create({
    data: {
      email,
      passwordHash,
      role: role.toUpperCase(),
    },
  });

  console.log('✅ User created:', {
    id: user.id,
    email: user.email,
    role: user.role,
  });

  await prisma.$disconnect();
}

createUser().catch(console.error);
```

Run it:
```powershell
node create-user.js user@example.com theirpassword USER
```

### Option 4: Direct Database Access

If you have direct access to PostgreSQL:

```sql
-- First, generate a bcrypt hash (use Node.js or online tool)
-- Example hash for password "password123":
-- $2b$10$rOzJqJqJqJqJqJqJqJqJuuJqJqJqJqJqJqJqJqJqJqJqJqJqJqJqJqJq

INSERT INTO "User" (id, email, "passwordHash", role, "createdAt", "updatedAt")
VALUES (
  'clx1234567890', -- Generate a CUID
  'user@example.com',
  '$2b$10$rOzJqJqJqJqJqJqJqJqJuuJqJqJqJqJqJqJqJqJqJqJqJqJqJqJqJqJq', -- Bcrypt hash
  'USER',
  NOW(),
  NOW()
);
```

## Security Notes

- Always use strong passwords (minimum 8 characters, mix of letters, numbers, symbols)
- Passwords are hashed using bcrypt (10 rounds)
- Never store plain text passwords
- Regular users should have `role: 'USER'`
- Only trusted administrators should have `role: 'ADMIN'`

## Testing

After creating a user, test login:
1. Go to http://localhost:3000/login
2. Enter the email and password you just created
3. You should be redirected to the account page

