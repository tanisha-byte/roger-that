# Roger That

A deterministic evaluation gauntlet for AI agents, built on air traffic control radiotelephony. Every clearance has a mandatory, enumerable readback (ICAO Doc 4444 / Doc 9432) — so grading is a parsing problem, not a judgment problem. No LLM judge anywhere in the scoring path.

**No mock agents.** Every session on the leaderboard is a real graded phone call to a real Bolna voice agent, or (with a key configured) a real campaign against a real LLM.

Full design: [`roger-that-project-brief.md`](roger-that-project-brief.md). Bug-hunting history: [`HISTORY.md`](HISTORY.md).

## What it does

- **Scorer** (`scorer/`) — normalizes ATC number-speak and NATO alphabet, extracts typed clearance items, grades readbacks/traps/hearback-errors/state-probes/emergencies, and rolls it into a weighted score. ≥99% agreement on a hand-labeled golden set.
- **Scenario engine** (`scenarios/`) — 6 seeded, randomized scenarios (descent, rapid-fire, callsign trap, hearback error, state probe, emergency); nothing is answerable by memorization.
- **Live Bolna calls** (`run_live_call.py`) — places a real outbound call to a real Bolna voice agent, waits for it to finish, pulls the transcript, matches every line back to the right scenario turn (any order), and grades it with the same scorer.
- **Dashboard** (`dashboard/index.html`, one FastAPI process) — two tabs:
  - **Flight Sim**: a pixel-art radar scene that replays any real session turn by turn — real ATC line, real pilot reply, a coin pop for a clean readback, a crash + lost life for a caught safety violation.
  - **Metrics**: leaderboard, learning curve, skill-diff viewer, full transcript replay, and a real-campaign launcher.
- **Also included**: an `OpenAICompatPilot` adapter for any real LLM, a Hermes learning-loop adapter (debrief + skill-diffing, contract-complete but untested against a live gateway), a radio-channel DSP chain, and an Atropos-style RL env wrapper.

## Quickstart

```bash
cd roger-that
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m pytest tests/ -q      # full test suite
uvicorn api:app --reload        # dashboard at http://localhost:8000
```

The database starts empty. Put real data in it one of two ways:

```bash
# a real phone call to a real Bolna agent (needs BOLNA_API_KEY)
python run_live_call.py --scenario callsign-trap-01 --seed 42 \
    --agent-id <your-bolna-agent-id> --to <e164-phone-number>
```

```python
# a real campaign against any OpenAI-compatible model (needs OPENAI_API_KEY)
from agents.openai_compat import OpenAICompatPilot
from orchestrator import run_campaign, list_scenarios
from db.store import Store

store = Store()
scenarios = list_scenarios()
run_campaign(store=store, campaign_id="eval", agent_name="gpt-4o-mini",
    agent_factory=lambda i: OpenAICompatPilot(model="gpt-4o-mini"),
    train_scenarios=scenarios[:-1], held_out_scenarios=scenarios[-1:],
    n_sessions=20, base_seed=42)
```
...or from the dashboard's Metrics tab, once a key is set in the environment `uvicorn` runs in.

## Layout

```
roger-that/
├── scenarios/       # YAML scenario packs
├── controller/       # deterministic controller + phraseology spelling
├── scorer/           # normalizer, grammar, grader
├── state/            # aircraft state tracker
├── agents/           # PilotAgent protocol + openai_compat, hermes, websocket_voice
├── orchestrator/      # scenario rendering, session runner, campaigns, live_bolna
├── loop/              # Hermes debrief protocol, leakage lint, skill snapshotting
├── channel/           # radio DSP chain, pluggable TTS/ASR
├── db/, export/       # sqlite storage, JSONL/Atropos exporters
├── dashboard/          # Flight Sim + Metrics SPA
├── api.py             # FastAPI (API + dashboard, one port)
├── run_live_call.py   # CLI: place a real Bolna call, score it, save it
└── tests/             # golden tests + integration tests
```

## Extending

- **New scenario**: drop a YAML file into `scenarios/` — picked up automatically. Schema documented in `orchestrator/scenario_loader.py`.
- **New agent**: implement `PilotAgent` (`agents/base.py`) and pass an `agent_factory` to `run_campaign`. Implement `LearningAgent` too if it should get debriefed and have its skill files snapshotted.
