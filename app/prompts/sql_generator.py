# app/prompts/sql_generator.py
"""Prompts for SQL generation and Data Explanation."""
from __future__ import annotations
from app.domain.models import ColumnSchema

_SQL_SYSTEM = """You are an expert SQLite Data Analyst.
Your task is to translate a user's natural language request (which may be in Arabic or English) into a valid SQLite SELECT query.

You will be provided with:
1. The Database Schema (Table name and Columns).
2. The User's Request.

RULES:
- Output STRICT JSON only.
- If the user asks a question that CANNOT be answered using the provided schema, set "can_answer" to false.
- If the user asks to modify data (INSERT, UPDATE, DELETE, DROP), set "can_answer" to false.
- ALWAYS append `LIMIT 100` to the query to prevent memory overload, unless the user explicitly asks for all data.
- If a column name has spaces or special characters, wrap it in double quotes, e.g., "Date of birth" or "First Name".
- SQLite date functions are different from other SQLs. Use `strftime` if needed.

JSON Format:
{
  "can_answer": true | false,
  "sql": "SELECT ...", 
  "error_reason": "If can_answer is false, explain why."
}
"""

_EXPLAIN_SYSTEM = """You are a helpful Data Analyst.
You will be provided with the user's original question, the SQL query executed, and the resulting data.
Your task is to explain the results in a clear, natural language summary (2-3 sentences).

RULES:
- If the user asked in Arabic, reply in Arabic.
- If the user asked in English, reply in English.
- Be concise and focus on the insights (e.g., "The highest sales were in Electronics").
"""


def build_sql_prompt(user_text: str, table_name: str, schema: list[ColumnSchema]) -> list[dict]:
    schema_str = "\n".join([f"- {col.name} ({col.type})" for col in schema])

    user_message = f"""Table Name: {table_name}
Schema:
{schema_str}

User Request: {user_text}
"""
    return [
        {"role": "system", "content": _SQL_SYSTEM},
        {"role": "user", "content": user_message}
    ]


def build_explanation_prompt(user_text: str, sql: str, result_data: list[dict]) -> list[dict]:
    data_str = str(result_data[:10])
    if len(result_data) > 10:
        data_str += f"\n...and {len(result_data) - 10} more rows."

    user_message = f"""User Question: {user_text}
SQL Executed: {sql}
Result Data:
{data_str}
"""
    return [
        {"role": "system", "content": _EXPLAIN_SYSTEM},
        {"role": "user", "content": user_message}
    ]