#!/bin/sh
# Script to check database connection and manually create admin user if needed

echo "🔍 Checking database connection..."
echo "DATABASE_URL: $DATABASE_URL"

# Check if we can connect to the database
npx prisma db execute --stdin <<EOF
SELECT 1;
EOF

if [ $? -eq 0 ]; then
  echo "✅ Database connection successful"
else
  echo "❌ Database connection failed"
  exit 1
fi

echo ""
echo "🔍 Checking if users table exists..."
npx prisma db execute --stdin <<EOF
SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'User';
EOF

echo ""
echo "🔍 Checking existing users..."
npx prisma db execute --stdin <<EOF
SELECT email, role FROM "User" LIMIT 5;
EOF

echo ""
echo "🌱 Attempting to seed admin user..."
if [ -n "$ADMIN_EMAIL" ] && [ -n "$ADMIN_PASSWORD" ]; then
  echo "Using ADMIN_EMAIL: $ADMIN_EMAIL"
  npx ts-node --project prisma/tsconfig.json prisma/seed.ts
else
  echo "❌ ADMIN_EMAIL or ADMIN_PASSWORD not set"
  exit 1
fi



