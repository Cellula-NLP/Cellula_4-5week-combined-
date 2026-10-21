"""SQLite implementation of the DatabaseManager."""
from __future__ import annotations

import logging
import os
import re
import sqlite3
from typing import Any

import pandas as pd
import sqlglot
from sqlglot import exp

from app.core.config import get_settings
from app.core.exceptions import DatasetError, SQLError
from app.domain.models import ColumnSchema
from app.domain.ports import DatabaseManager

logger = logging.getLogger(__name__)


class SQLiteManager(DatabaseManager):
    """Manages SQLite database operations for uploaded datasets."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._db_path = os.path.join(self._settings.data_dir, "analytics.db")
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_conn():
            pass

    def create_table_from_csv(self, csv_path: str, table_name: str) -> tuple[str, int, list[ColumnSchema]]:
        if not os.path.exists(csv_path):
            raise DatasetError(f"File not found: {csv_path}")

        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                raise DatasetError("The uploaded CSV file is empty.")

            # Sanitize table name
            table_name = re.sub(r'\W+', '_', table_name).strip('_').lower()
            if not table_name:
                table_name = "uploaded_data"

            # Sanitize column names (replace spaces and special chars with underscores)
            df.columns = [re.sub(r'\W+', '_', str(col)).strip('_').lower() for col in df.columns]

            with self._get_conn() as conn:
                conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
                df.to_sql(table_name, conn, index=False, if_exists="replace")

            logger.info("Successfully created table '%s' with %d rows.", table_name, len(df))

            schema = self.get_table_schema(table_name)
            return table_name, len(df), schema

        except pd.errors.EmptyDataError:
            raise DatasetError("The CSV file is empty or malformed.")
        except Exception as exc:
            logger.error("Failed to process CSV: %s", exc)
            raise DatasetError(f"Failed to process dataset: {exc}") from exc

    def get_table_schema(self, table_name: str) -> list[ColumnSchema]:
        schema: list[ColumnSchema] = []
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute(f'PRAGMA table_info("{table_name}")')
                rows = cursor.fetchall()

                for row in rows:
                    schema.append(ColumnSchema(name=row["name"], type=row["type"]))
            return schema
        except Exception as exc:
            logger.error("Failed to fetch schema for %s: %s", table_name, exc)
            raise DatasetError(f"Could not read table schema: {exc}") from exc

    def execute_query(self, query: str) -> list[dict[str, Any]]:
        try:
            parsed = sqlglot.parse(query, read="sqlite")
            if not parsed:
                raise SQLError("Empty SQL query.")

            for statement in parsed:
                if not isinstance(statement, exp.Select):
                    raise SQLError("Security Violation: Only SELECT queries are allowed.")

                if statement.find(exp.Drop) or statement.find(exp.Delete) or statement.find(exp.Update):
                    raise SQLError("Security Violation: Destructive commands detected.")

        except sqlglot.errors.ParseError as exc:
            raise SQLError(f"Invalid SQL syntax: {exc}") from exc

        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error as exc:
            logger.error("SQLite execution error: %s", exc)
            raise SQLError(f"Database execution failed: {exc}") from exc