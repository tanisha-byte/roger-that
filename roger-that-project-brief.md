# Roger That

**A deterministic evaluation gauntlet for AI agents, built on air traffic control radiotelephony.**

Voice and chat agents are graded today by having one LLM judge another. ATC phraseology is defined by regulation (ICAO Doc 4444 / Doc 9432, Annex 10 Vol II): every clearance has a mandatory readback whose required items are enumerable and machine-checkable. Roger That simulates a controller, degrades the radio channel, throws safety traps at the agent-under-test, and grades every response with a parser — not a judge.

One harness, three products:

1. **Benchmark** — a leaderboard of models / ASR+LLM+TTS combos scored on readback accuracy, safety violations, and minimum survivable radio quality.
2. **Learning-loop testbed** — Hermes Agent (Nous Research) runs repeated sessions with only its scorecards as feedback; the score-vs-session curve measures whether its self-improvement loop actually works.
3. **RL environment** — every session emits a (trajectory, scalar reward) pair with a verifiable, regulation-defined reward, exportable to Atropos for fine-tuning.

---

## 1. The Problem

Conversational agent evaluation has no ground truth. "Was that a good response?" is answered by another LLM, and everyone knows the grades are mush — biased, unstable across judge versions, and impossible to defend. Meanwhile, teams shipping voice agents regression-test them by manually calling their own bot.

Aviation solved the grading problem eighty years ago. When a controller says:

> "VT-ABC, descend flight level 80, QNH 1013"

the pilot's readback **must** contain the flight level, the QNH, and the callsign. Not "should" — must, by regulation, in a defined structure. Miss the QNH and it is a hearback/readback error, a formally tracked incident category in real aviation safety data. This means:

- **Grading is a parsing problem**, not a judgment problem.
- **Failure modes are catastrophic-but-testable**: accepting a clearance meant for a similar callsign, missing a controller's hearback error, botching an emergency call.
- **Difficulty is tunable on a physical axis**: radio channel quality (SNR), which stresses ASR in a controlled, reportable way.

## 2. What It Does

- Runs an **agent-under-test as the pilot** in scripted scenarios, over text or a simulated voice channel.
- A **scripted controller** issues clearances; a **fault injector** adds traps (callsign confusion, hearback errors, rapid-fire multi-item clearances, emergencies, frequency clutter).
- A **radio channel simulator** degrades controller audio (300–3400 Hz bandpass, static bursts, clipping, squelch tails) at configurable SNR tiers.
- A **deterministic phraseology scorer** parses every pilot transmission against the expected readback grammar and the live aircraft state.
- A **session orchestrator** runs campaigns: N sessions × M agents × K noise tiers, with train/held-out scenario splits.
- A **Hermes adapter** closes the learning loop: post-session debrief messages in, skill-file snapshots out.
- A **dashboard** renders per-turn scorecards, per-agent learning curves, and the combo leaderboard.
- An **exporter** writes scored trajectories in Atropos-compatible format.

## 3. Non-Goals (v1)

- No flight physics, no radar, no airspace simulation. The "aircraft" is a state dict.
- No full ATC domain coverage. v1 is en-route/approach clearances + one emergency type. Ground/taxi phraseology is a later scenario pack.
- No attempt to certify anything for real aviation use. This is an AI eval tool that borrows aviation's grading rules; it is not a pilot training device.
- No LLM judge anywhere in the scoring path. (An optional LLM "style commentary" can exist in the dashboard, clearly separated and never counted.)

---

## 4. System Architecture

```
                          ┌──────────────────────────────────────────────┐
                          │              Session Orchestrator          │
                          │  campaigns, seeds, train/held-out splits   │
                          └───────┬──────────────────────────────┬───────┘
                                  │                           │
                                  ▼                           ▼
   ┌────────────────────┐   ┌────────────────┐            ┌────────────────────┐
   │  Scenario Engine ├──▶│  Controller  │            │  Agent Adapters  │
   │  YAML + faults + │   │  (scripted   │            │  hermes / openai │
   │  randomization   │   │   state m/c) │            │  bolna / generic │
   └────────────────────┘   └──────┬──────────┘            └──────────┬──────────┘
                                 │ text                        │
                                 ▼                             │
                        ┌───────────────────┐   audio   ┌──────────┼───────────┐
                        │  Voice Channel  ├──────────▶│  Agent under    │
                        │  TTS → radio    │           │  test (pilot)   │
                        │  degrade → ASR  │◀──────────────│                 │
                        └───────────────────┘   audio   └──────────┬──────────┘
                                 │  (voice mode only)          │ readback (text)
                                 ▼                             ▼
   ┌────────────────────┐   ┌──────────────────────────────────────────────┐
   │  Aircraft State  ├──▶│         Phraseology Scorer               │
   │  altitude, hdg,  │   │  normalize → callsign gate → item        │
   │  clearances, freq│   │  extraction → readback grade → safety    │
   └────────────────────┘   └──────────────────────┬───────────────────────┘
                                              │ turn records + scorecards
                                              ▼
                          ┌──────────────────────────────────────────────┐
                          │              SQLite metrics DB             │
                          └───────┬──────────────────┬──────────────────┘
                                  ▼                  ▼
                       ┌──────────────────┐   ┌──────────────────────┐
                       │  FastAPI +     │   │  Exporters           │
                       │  Dashboard     │   │  Atropos / JSONL     │
                       └──────────────────┘   └──────────────────────┘

   Hermes learning loop (sidecar):
   scorecard ──▶ debrief message ──▶ Hermes skill loop ──▶ skill snapshot diff
```

**Two transport modes, one scoring path.** In text mode the controller line goes straight to the agent and the reply comes back as text. In voice mode the same line is synthesized, degraded, and sent as audio; the agent's audio reply is transcribed by a reference ASR before scoring. The scorer never knows which mode produced the text — mode is just a column in the DB.

---

## 5. Components

### 5.1 Scenario Engine (`scenarios/`)

- Scenarios are YAML files: a script of controller turns, expected readback items per turn, and fault injections.
- **Parameterization**: callsigns, flight levels, headings, QNH, frequencies are template variables regenerated from a seeded RNG every session. Nothing is answerable by memorization; only the *procedure* transfers.
- **Fault types (v1)**: `similar_callsign_trap`, `hearback_error`, `rapid_fire` (3–5 readback items in one transmission), `emergency` (engine failure → expected MAYDAY elements), `frequency_clutter` (other-pilot chatter the agent must ignore), `nonstandard_phraseology` (rambling GA-style controller lines).

```yaml
id: descent-with-confusion-01
difficulty: 2
airport: VOBL
aircraft:
  callsign: "{random_callsign}"          # e.g. VT-ABC
  intruder: "{similar_to: callsign}"     # e.g. VT-ADC
script:
  - atc: "{callsign}, descend flight level {fl}, QNH {qnh}"
    expect_readback: [fl, qnh, callsign]
  - atc: "{intruder}, turn left heading {hdg}"
    expect: no_response                   # the trap
  - atc: "{callsign}, contact approach {freq}"
    expect_readback: [freq, callsign]
faults:
  - type: hearback_error
    turn: 3
    corrupt: freq                         # controller reads back wrong freq
```

### 5.2 Controller Agent (`controller/`)

- A **scripted state machine**, not an LLM. Determinism in the controller is what keeps the eval reproducible.
- Advances through the scenario script; branches on agent behavior (e.g., if the agent misses a readback item, the controller issues the standard "readback correct/negative" exchange as real controllers do — this branch is itself scored).
- Optional "clutter voices" (other pilots on frequency) are canned audio/text lines, also scripted.

### 5.3 Voice Channel (`channel/`)

- **TTS**: pluggable (any provider, or Hermes's TTS tool); controller voice fixed per campaign for comparability.
- **Radio degradation DSP chain** (Python, `scipy`/`pedalboard`): 300–3400 Hz bandpass → companding/clipping → additive static at target SNR → squelch tail on transmission end. SNR tiers: clean / 20 dB / 15 dB / 10 dB / 8 dB / 5 dB.
- **Reference ASR** for the agent's replies in voice mode (fixed, documented model), so agent-side scoring is not confounded by the harness's own transcription. Voice-stack benchmarks additionally record the agent's *own* pipeline latencies.

### 5.4 Agent Adapters (`agents/`)

A minimal interface every adapter implements:

```python
class PilotAgent(Protocol):
    def start_session(self, role_prompt: str) -> None: ...
    def on_transmission(self, audio_or_text) -> Reply: ...
    def end_session(self) -> None: ...
```

- **`agents/hermes.py`** — drives a fresh Hermes subagent per session over its CLI/gateway; supports the debrief channel and skill-directory snapshotting (see §7).
- **`agents/openai_compat.py`** — any chat-completions endpoint (covers most models via OpenRouter).
- **`agents/websocket_voice.py`** — generic audio-in/audio-out websocket, suitable for Bolna-style voice pipelines.

The role prompt is deliberately minimal and identical for all agents: *"You are the pilot of {callsign} on this frequency. Respond to ATC."* No phraseology instruction — knowing (or learning) the rules is part of the test.

### 5.5 Aircraft State Tracker (`state/`)

A per-session dict: current/assigned altitude, heading, speed, QNH, active frequency, open clearances. Updated only when the scorer confirms a clearance was correctly accepted. Enables **consistency checks**: a later turn like "confirm assigned level" is graded against state, which is how multi-turn memory gets tested without any judge.

### 5.6 Phraseology Scorer (`scorer/`) — the crown jewel

Pipeline per pilot transmission:

1. **Normalization** — lowercase, expand aviation number-speak ("one zero one three" → 1013, "flight level eight zero" → FL80, decimal "one one niner decimal seven" → 119.7), map letter callsigns through the NATO alphabet, tolerate common ASR confusions via a fixed substitution table (documented, versioned).
2. **Callsign gate** — was this transmission addressed to us? Did the reply carry our callsign? Did the agent respond to the intruder's clearance? (`responded_to_intruder_clearance` is the headline safety violation.)
3. **Item extraction** — a small grammar (hand-rolled recursive-descent or `lark`) that pulls typed items out of the readback: `FL(int)`, `HDG(int)`, `QNH(int)`, `FREQ(float)`, `RWY(str)`, `SQUAWK(int)`.
4. **Readback grading** — set-compare extracted items against `expect_readback` for the turn: item present, value correct, callsign appended. Partial credit per item.
5. **Safety detection** — cross-checks: accepted-wrong-clearance, missed hearback error (controller's corrupted repeat not challenged), state inconsistency, missing MAYDAY elements in emergencies (callsign, nature, intentions, position).

Output per turn is a structured record; per session, an aggregate scorecard:

```json
{
  "session": 7, "agent": "hermes-v1", "mode": "text", "snr": null,
  "score": 0.62,
  "readback_completeness": 0.71,
  "safety_violations": ["responded_to_intruder_clearance"],
  "hearback_errors_caught": "0/1",
  "turns": [
    {"turn": 1, "items": {"fl": "ok", "qnh": "missing", "callsign": "ok"}, "latency_ms": 1400},
    {"turn": 2, "violation": "responded_to_intruder_clearance"},
    {"turn": 3, "hearback_error_caught": false}
  ]
}
```

**Score = weighted composite** (weights in one config file, versioned):
readback completeness 40% · safety (violations are large negative) 35% · state consistency 15% · emergency handling 10%. Latency reported separately, never blended in — correctness and speed are different leaderboards.

### 5.7 Session Orchestrator (`orchestrator/`)

- Runs **campaigns**: `(agent, scenario_pool, snr_tier, n_sessions, seed)`.
- Enforces **train/held-out split**: debriefs are sent only for training-pool sessions; held-out sessions are scored silently.
- Handles retries, timeouts (a non-response within 10 s of transmission end = scored miss), and full transcript + audio archival for replay.

### 5.8 Hermes Learning-Loop Adapter (`agents/hermes.py` + `loop/`)

- **Debrief protocol**: after each training session, send the scorecard to Hermes as a plain chat message, *observational only* — what was missed, never why or how to fix it. A `debrief_leakage` lint checks debrief strings against a banned-phrase list ("must read back", "ICAO requires", …) so the harness never teaches.
- **Skill archaeology**: snapshot Hermes's skill directory (agentskills.io-format files) after every session; store diffs. The evolution of its self-written `atc-phraseology` skill is a first-class artifact in the dashboard.
- **Injection boundary** (honesty rule): the harness never writes, edits, or seeds skill files. Hermes may use its own web search to find ICAO rules — that's fair game and part of the result.

### 5.9 Storage (`db/`)

SQLite, three tables: `sessions`, `turns`, `skill_snapshots`. Same reasoning as Bolna-Monitor: at eval volumes this is thousands of rows per campaign; indexed SQLite is trivially sufficient, and the query layer is swappable for Postgres.

### 5.10 Dashboard + API (`api.py`, `dashboard/`)

FastAPI + a single-page Chart.js dashboard (no build step):

```
GET  /api/leaderboard?metric=score&snr=15       agents ranked, per noise tier
GET  /api/agents/{id}/curve                     score vs session (train + held-out)
GET  /api/sessions/{id}                         full scorecard + transcript replay
GET  /api/sessions/{id}/audio/{turn}            degraded controller audio (demo gold)
GET  /api/skills/{agent}/timeline               skill-file diff viewer
POST /api/campaigns                             launch a campaign
```

Headline visuals: the **learning curve** (Hermes full-loop vs memory-wiped baseline, held-out overlay), the **SNR survival chart** (score vs noise tier per combo — "minimum survivable radio quality"), and the per-turn scorecard with red safety flags.

### 5.11 Exporters (`export/`)

- **JSONL trajectories**: `(role_prompt, transmissions[], replies[], per_turn_scores[], session_reward)`.
- **Atropos environment wrapper**: exposes a session as `reset()/step()` with the composite score as reward, so Nous-stack RL runs against Roger That directly.

---

## 6. Scoring Rubric (v1 metric definitions)

| Metric | Definition |
|---|---|
| Readback completeness | Fraction of mandatory items correctly read back, value-exact, across all turns |
| Callsign discipline | Own callsign appended on every reply; zero responses to other callsigns |
| Wrong-clearance acceptance | Count of intruder clearances acted on (each is a hard penalty) |
| Hearback catch rate | Injected controller errors challenged / injected |
| State consistency | "Confirm"-type probes answered correctly from tracked state |
| Emergency completeness | MAYDAY elements present when scripted (callsign, nature, intentions, position) |
| Radio robustness | Highest SNR tier at which composite score stays ≥ 0.8 |
| Response latency | End-of-transmission → first reply token/audio (reported, not blended) |

## 7. Experiment Design (the Hermes result)

- **Run A** — Hermes, memory + skill loop on, debriefs on training pool.
- **Run B** — same underlying model, memory wiped every session (ablation: is improvement from the loop or from nothing?).
- **Run C** *(optional ceiling)* — Hermes seeded with a human-written ICAO skill on day one.
- 30 training scenarios / 10 held-out, 50 sessions per run, 3 seeds. Values randomized per session.
- **Claims the data can support**: (1) the loop works iff A's held-out curve rises above B's; (2) generalization iff held-out tracks training; (3) skill archaeology shows *how* — from crude "repeat all numbers" heuristics to a mandatory-items checklist.

## 8. Scenario Catalog v1 (10 scenarios)

1. Simple descent clearance (calibration baseline)
2. Rapid-fire: heading + altitude + speed + QNH in one transmission
3. Similar-callsign trap (the demo scenario)
4. Hearback error: controller corrupts the frequency on repeat
5. Frequency change chain with "confirm assigned level" state probe
6. Emergency: engine failure mid-scenario → MAYDAY grading
7. Conditional clearance ("after passing FL100, reduce speed 220 knots")
8. Frequency clutter: three other aircraft active, agent addressed twice
9. Nonstandard controller phrasing (rambling, still valid instructions)
10. Amended clearance (climb, then re-cleared lower → tests state overwrite)

## 9. Repository Layout

```
roger-that/
├── scenarios/            # YAML scenario packs
├── controller/           # scripted controller state machine
├── channel/              # TTS glue + radio DSP + reference ASR
├── agents/               # hermes.py, openai_compat.py, websocket_voice.py
├── state/                # aircraft state tracker
├── scorer/               # normalizer, grammar, grader, safety checks
├── orchestrator/         # campaigns, splits, seeds
├── loop/                 # debrief protocol, skill snapshotting, leakage lint
├── db/                   # sqlite schema + queries
├── export/               # jsonl + atropos wrapper
├── dashboard/            # single-file Chart.js SPA
├── api.py                # FastAPI (API + dashboard, one port)
├── tests/                # scorer golden tests + seed_demo_data.py
└── README.md
```

## 10. Tech Stack

- **Python 3.11+, FastAPI, SQLite** — same zero-external-deps philosophy as Bolna-Monitor.
- **DSP**: `scipy.signal` (bandpass, noise mix), optionally `pedalboard` for the compressor/clipper.
- **Parsing**: `lark` or hand-rolled recursive descent; the grammar is small.
- **TTS/ASR**: pluggable providers; Hermes's own TTS tool works for the controller voice in Hermes runs.
- **Dashboard**: Chart.js via CDN, no build step.
- **Datasets/refs**: ICAO Doc 4444 & Doc 9432 (phraseology + readback rules), ATCO2 and ATCOSIM corpora (real controller-pilot audio for validating the normalizer and building realistic clutter), LiveATC archives (demo material).

## 11. Milestones

**Phase 1 — Scorer + text mode (week 1–2).** Normalizer, grammar, grader, golden test suite of 100 hand-labeled transmissions (target: ≥99% agreement with hand labels — the scorer must be more trustworthy than any agent it grades). Scenario engine with randomization, scripted controller, openai-compat adapter, SQLite, seed_demo_data.py. *Exit: run scenario 1–5 against two models, scorecards in DB.*

**Phase 2 — Dashboard + traps (week 3).** SPA with leaderboard, session replay, red-flag scorecards. Scenarios 6–10 incl. callsign trap and hearback fault. *Exit: the viral demo — play the trap, show the red flag.*

**Phase 3 — Hermes loop (week 4–5).** Hermes adapter, debrief protocol + leakage lint, skill snapshots, Runs A/B, learning-curve chart with held-out overlay. *Exit: the headline chart.*

**Phase 4 — Voice + RL (week 6+).** Radio DSP chain, SNR tiers, survival chart, websocket voice adapter (Bolna-compatible), Atropos exporter. *Exit: minimum-survivable-radio leaderboard + exportable RL environment.*

## 12. Risks & Mitigations

- **Scorer false negatives on valid phrasing variants** (readbacks have legal word-order flexibility) — grade on extracted item *sets*, not templates; grow the golden test suite from every disputed case; version the normalizer.
- **ASR confusion unfairly penalizing agents in voice mode** — fixed reference ASR, per-tier reporting, and publish the substitution table; text-mode scores always shown alongside.
- **Debrief accidentally teaching** — leakage lint + a "zero-debrief" control campaign if curves look suspicious.
- **Hermes memorizing despite randomization** — held-out pool is the backstop; also rotate airport/frequency bands between train and held-out.
- **Scope creep toward flight sim** — the non-goals section is load-bearing; aircraft stays a dict.

## 13. Demo Script (3 minutes)

1. Play the SNR-8dB controller audio — audience can barely parse it. Agent reads back all four items perfectly. Scorecard: green.
2. Same session, two turns later: controller clears **VT-ADC** to descend. Agent (as VT-ABC) reads it back and complies. Scorecard: red — `responded_to_intruder_clearance`. "This is the error category that has killed people; your voice agent just made it."
3. Cut to the Hermes learning curve: session 1 score 41%, session 30 score 88%, held-out tracking. Open the skill diff viewer: the skill it wrote itself at session 4 vs session 25.
4. Close: "Every grade you saw came from a parser and a regulation, not a judge."
