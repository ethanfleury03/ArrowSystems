/* eslint-disable no-console */
import { PrismaClient, UserRole } from '@prisma/client';
import bcrypt from 'bcryptjs';
import 'dotenv/config';

const prisma = new PrismaClient();

async function main() {
  const {
    SEED_ADMIN_EMAIL,
    SEED_ADMIN_PASSWORD,
    SEED_TECH_EMAIL,
    SEED_TECH_PASSWORD,
  } = process.env;

  if (!SEED_ADMIN_EMAIL || !SEED_ADMIN_PASSWORD || !SEED_TECH_EMAIL || !SEED_TECH_PASSWORD) {
    throw new Error('Seed environment variables missing. Please set SEED_ADMIN_EMAIL, SEED_ADMIN_PASSWORD, SEED_TECH_EMAIL, SEED_TECH_PASSWORD.');
  }

  const adminHash = await bcrypt.hash(SEED_ADMIN_PASSWORD, 10);
  const techHash = await bcrypt.hash(SEED_TECH_PASSWORD, 10);

  await prisma.user.upsert({
    where: { email: SEED_ADMIN_EMAIL },
    update: {},
    create: {
      email: SEED_ADMIN_EMAIL,
      name: 'Administrator',
      role: UserRole.ADMIN,
      passwordHash: adminHash,
    },
  });

  await prisma.user.upsert({
    where: { email: SEED_TECH_EMAIL },
    update: {},
    create: {
      email: SEED_TECH_EMAIL,
      name: 'Technician',
      role: UserRole.TECHNICIAN,
      passwordHash: techHash,
    },
  });

  console.log('✅ Seed data inserted');
}

main()
  .catch((e) => {
    console.error(e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });


