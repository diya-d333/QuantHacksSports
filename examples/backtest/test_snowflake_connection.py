import os
from pathlib import Path

import snowflake.connector
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

with snowflake.connector.connect(
    account=os.environ["SNOWFLAKE_ACCOUNT"],
    user=os.environ["SNOWFLAKE_USER"],
    authenticator="OAUTH_AUTHORIZATION_CODE",
    warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
    database=os.environ["SNOWFLAKE_DATABASE"],
    schema=os.environ["SNOWFLAKE_SCHEMA"],
) as connection:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT CURRENT_DATABASE(), CURRENT_SCHEMA(), CURRENT_USER()"
        )
        database, schema, user = cursor.fetchone()
        print(f"Connected to Snowflake! Database: {database}")
        print(f"Schema: {schema} | User: {user}")
        cursor.execute("SELECT CURRENT_ROLE()")
        print(f"Role: {cursor.fetchone()[0]}")