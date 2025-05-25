import psycopg2
import os
from urllib.parse import urlparse

# Get database connection parameters from environment variables or use defaults
PG_USER = os.environ.get("POSTGRES_USER", "postgres")
PG_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "postgres")
PG_HOST = os.environ.get("POSTGRES_HOST", "you_must_enter_a_postgres_host")
PG_PORT = os.environ.get("POSTGRES_PORT", "5432")
PG_DB = os.environ.get("POSTGRES_DB", "optuna")

# Construct the database URL
db_url = f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"

parsed = urlparse(db_url)

# Extract connection info
user = parsed.username
password = parsed.password
host = parsed.hostname
port = parsed.port or 5432
database = parsed.path.lstrip('/')

# Connect to the target database
conn = psycopg2.connect(
    dbname=database,
    user=user,
    password=password,
    host=host,
    port=port,
    sslmode="require"
)
conn.autocommit = True
cur = conn.cursor()

# Disable triggers and constraints, then truncate all tables
cur.execute("""
    DO $$
    DECLARE
        r RECORD;
    BEGIN
        FOR r IN (SELECT tablename FROM pg_tables WHERE schemaname = 'public') LOOP
            EXECUTE 'TRUNCATE TABLE public.' || quote_ident(r.tablename) || ' RESTART IDENTITY CASCADE;';
        END LOOP;
    END
    $$;
""")
print(f"All data in database '{database}' deleted.")

cur.close()
conn.close()