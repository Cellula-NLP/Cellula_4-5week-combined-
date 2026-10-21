# app/services/sql_service.py
"""Orchestrates Text -> SQL -> DB Execution -> Explanation."""
from __future__ import annotations

import json
import logging

from app.core.exceptions import LLMError, SQLError
from app.domain.ports import LLMProvider, DatabaseManager
from app.prompts.sql_generator import build_sql_prompt, build_explanation_prompt

logger = logging.getLogger(__name__)


class SQLService:
    def __init__(self, llm: LLMProvider, db: DatabaseManager) -> None:
        self._llm = llm
        self._db = db

    async def analyze(self, user_text: str, table_name: str) -> dict:
        """Main entry point for the analysis endpoint."""

        # 1. Fetch Schema
        try:
            schema = self._db.get_table_schema(table_name)
        except Exception as exc:
            raise SQLError(f"Could not load table schema: {exc}")

        # 2. Generate SQL via LLM
        messages = build_sql_prompt(user_text, table_name, schema)
        try:
            raw_json = await self._llm.complete(messages, json_mode=True, temperature=0.0)
            sql_data = self._parse_json(raw_json)
        except Exception as exc:
            logger.error("Failed to generate SQL: %s", exc)
            raise LLMError("Failed to understand the request. Please try again.") from exc

        # 3. Check if the LLM can answer
        if not sql_data.get("can_answer", False):
            reason = sql_data.get("error_reason", "I cannot answer this based on the provided data.")
            return {
                "user_text": user_text,
                "generated_sql": "",
                "result_data": [],
                "explanation": reason
            }

        sql_query = sql_data.get("sql", "")
        if not sql_query:
            raise SQLError("LLM returned an empty SQL query.")

        # 4. Execute SQL (The DatabaseManager handles the security validation)
        try:
            result_data = self._db.execute_query(sql_query)
        except Exception as exc:
            logger.error("SQL Execution failed: %s", exc)
            raise SQLError(f"Database execution failed: {exc}")

        # 5. Generate Explanation
        explanation = ""
        if result_data:
            try:
                exp_messages = build_explanation_prompt(user_text, sql_query, result_data)
                explanation = await self._llm.complete(exp_messages, temperature=0.5)
            except Exception as exc:
                logger.warning("Failed to generate explanation: %s", exc)
                explanation = "Data retrieved successfully, but explanation generation failed."
        else:
            explanation = "The query executed successfully but returned no results."

        return {
            "user_text": user_text,
            "generated_sql": sql_query,
            "result_data": result_data,
            "explanation": explanation.strip()
        }

    @staticmethod
    def _parse_json(raw: str) -> dict:
        raw = raw.strip()
        if raw.startswith("```"):
            import re
            raw = re.sub(r"^```[a-zA-Z]*\n", "", raw)
            raw = re.sub(r"\n```$", "", raw)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            import re
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return {}