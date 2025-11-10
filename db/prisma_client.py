import os
from functools import lru_cache

from prisma import Prisma


@lru_cache(maxsize=1)
def get_prisma() -> Prisma:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL environment variable is required to start the backend.")
    return Prisma()


