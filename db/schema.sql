CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_id TEXT NOT NULL,
    agent TEXT NOT NULL,
    scenario_id TEXT NOT NULL,
    seed INTEGER NOT NULL,
    mode TEXT NOT NULL,             -- text | voice
    snr TEXT,                       -- null for text mode; snr tier label for voice mode
    pool TEXT NOT NULL,             -- train | held_out
    session_index INTEGER NOT NULL, -- position within this agent's run, for the learning curve
    score REAL NOT NULL,
    readback_completeness REAL NOT NULL,
    callsign_discipline REAL NOT NULL,
    safety_score REAL NOT NULL,
    safety_violations TEXT NOT NULL,   -- JSON list
    hearback_catch_rate REAL,
    state_consistency REAL NOT NULL,
    emergency_completeness REAL NOT NULL,
    weights_version TEXT NOT NULL,
    scorecard_json TEXT NOT NULL,      -- full scorecard, for the API to return verbatim
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_agent ON sessions(agent);
CREATE INDEX IF NOT EXISTS idx_sessions_campaign ON sessions(campaign_id);
CREATE INDEX IF NOT EXISTS idx_sessions_scenario ON sessions(scenario_id);

CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    turn_index INTEGER NOT NULL,
    kind TEXT NOT NULL,
    atc_text TEXT NOT NULL,
    reply_text TEXT NOT NULL,
    latency_ms INTEGER,
    record_json TEXT NOT NULL     -- full TurnRecord.to_dict()
);

CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);

CREATE TABLE IF NOT EXISTS skill_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent TEXT NOT NULL,
    session_index INTEGER NOT NULL,
    skill_file TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_skills_agent ON skill_snapshots(agent);
