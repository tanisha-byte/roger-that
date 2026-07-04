"""SQLite storage layer.

Same reasoning as any small eval harness: at eval volumes this is thousands
of rows per campaign; indexed SQLite is trivially sufficient, and every
query here is plain SQL so the layer is swappable for Postgres later without
touching callers.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA_PATH = Path(__file__).parent / "schema.sql"
DEFAULT_DB_PATH = Path(__file__).parent.parent / "roger_that.db"


class Store:
    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(SCHEMA_PATH.read_text())
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # -- writes ----------------------------------------------------------

    def insert_session(self, row: Dict[str, Any]) -> int:
        cur = self._conn.execute(
            """INSERT INTO sessions
               (campaign_id, agent, scenario_id, seed, mode, snr, pool, session_index,
                score, readback_completeness, callsign_discipline, safety_score,
                safety_violations, hearback_catch_rate, state_consistency,
                emergency_completeness, weights_version, scorecard_json, created_at)
               VALUES (:campaign_id, :agent, :scenario_id, :seed, :mode, :snr, :pool, :session_index,
                       :score, :readback_completeness, :callsign_discipline, :safety_score,
                       :safety_violations, :hearback_catch_rate, :state_consistency,
                       :emergency_completeness, :weights_version, :scorecard_json, :created_at)""",
            {**row, "safety_violations": json.dumps(row["safety_violations"])},
        )
        self._conn.commit()
        return cur.lastrowid

    def insert_turns(self, session_id: int, turns: List[Dict[str, Any]]) -> None:
        self._conn.executemany(
            """INSERT INTO turns (session_id, turn_index, kind, atc_text, reply_text, latency_ms, record_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [
                (session_id, t["turn_index"], t["kind"], t["atc_text"], t["reply_text"], t.get("latency_ms"), json.dumps(t["record"]))
                for t in turns
            ],
        )
        self._conn.commit()

    def insert_skill_snapshot(self, agent: str, session_index: int, skill_file: str, content: str, created_at: str) -> None:
        self._conn.execute(
            "INSERT INTO skill_snapshots (agent, session_index, skill_file, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (agent, session_index, skill_file, content, created_at),
        )
        self._conn.commit()

    # -- reads -------------------------------------------------------------

    def leaderboard(self, metric: str = "score", snr: Optional[str] = None) -> List[Dict[str, Any]]:
        allowed = {"score", "readback_completeness", "safety_score", "state_consistency", "emergency_completeness"}
        if metric not in allowed:
            metric = "score"
        query = f"""
            SELECT agent, mode, snr, COUNT(*) AS n_sessions, AVG({metric}) AS avg_metric,
                   AVG(score) AS avg_score, AVG(safety_score) AS avg_safety,
                   SUM(CASE WHEN safety_violations != '[]' THEN 1 ELSE 0 END) AS sessions_with_violations
            FROM sessions
        """
        params: List[Any] = []
        if snr is not None:
            query += " WHERE snr = ?"
            params.append(snr)
        query += " GROUP BY agent, mode, snr ORDER BY avg_metric DESC"
        rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def agent_curve(self, agent: str) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            """SELECT session_index, pool, score, scenario_id, created_at
               FROM sessions WHERE agent = ? ORDER BY session_index ASC""",
            (agent,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_session(self, session_id: int) -> Optional[Dict[str, Any]]:
        row = self._conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            return None
        session = dict(row)
        session["scorecard"] = json.loads(session.pop("scorecard_json"))
        session["safety_violations"] = json.loads(session["safety_violations"])
        turn_rows = self._conn.execute(
            "SELECT * FROM turns WHERE session_id = ? ORDER BY turn_index ASC", (session_id,)
        ).fetchall()
        turns = []
        for t in turn_rows:
            td = dict(t)
            td["record"] = json.loads(td.pop("record_json"))
            turns.append(td)
        session["turns"] = turns
        return session

    def list_sessions(self, campaign_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        query = "SELECT id, campaign_id, agent, scenario_id, mode, snr, pool, session_index, score, created_at FROM sessions"
        params: List[Any] = []
        if campaign_id:
            query += " WHERE campaign_id = ?"
            params.append(campaign_id)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def skill_timeline(self, agent: str) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT session_index, skill_file, content, created_at FROM skill_snapshots WHERE agent = ? ORDER BY session_index ASC",
            (agent,),
        ).fetchall()
        return [dict(r) for r in rows]

    def snr_survival(self) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            """SELECT agent, snr, AVG(score) as avg_score, COUNT(*) as n
               FROM sessions WHERE snr IS NOT NULL GROUP BY agent, snr ORDER BY agent, snr"""
        ).fetchall()
        return [dict(r) for r in rows]
