# Roger That

A deterministic evaluation gauntlet for AI agents, built on air traffic control radiotelephony. Grading is a parsing problem, not a judgment problem: every clearance has a mandatory, enumerable readback (ICAO Doc 4444 / Doc 9432), and this harness parses it — no LLM judge anywhere in the scoring path.

**No mock agents anywhere in this repository's runtime state.** Every session in `roger_that.db`, every leaderboard row, every session the dashboard's Flight Sim tab can play back is a real graded phone call to a real Bolna voice agent, or (once a key is configured) a real campaign against a real LLM. There used to be a `ScriptedMockPilot`/`LearningMockPilot` pair used to validate the harness's plumbing before any real target was available — it's gone; see "History" below for why it existed and why removing it was the right call once real data existed to replace it.

See [`roger-that-project-brief.md`](roger-that-project-brief.md) for the full original design. This is the implementation.

## What's fully working right now

- **Scorer** (`scorer/`): normalizer (ATC number-speak, NATO alphabet, ASR substitution table), item-extraction grammar, readback/trap/hearback/state-probe/emergency graders, weighted composite score. Golden tests, ≥99% item-agreement on the hand-labeled set (`tests/golden/transmissions.json`) — see `tests/test_scorer.py`.
- **Scenario engine** (`scenarios/*.yaml`, `orchestrator/scenario_loader.py`): 6 scenarios from the v1 catalog (simple descent, rapid-fire, callsign trap, hearback error, state-probe/frequency-chain, emergency), fully seeded/randomized per session — callsigns, flight levels, QNH, headings, frequencies never repeat, only the procedure transfers.
- **Scripted controller** (`controller/`): deterministic state machine + ICAO-style phraseology spelling (NATO alphabet, digit-by-digit numbers). Not an LLM — that's what keeps the eval reproducible.
- **Aircraft state tracker** (`state/`): per-session dict updated only on scorer-confirmed items, so "confirm assigned level"-style probes are graded against ground truth.
- **Session + campaign orchestrator** (`orchestrator/`): runs one scenario against one agent end to end, including the real ATC "readback incorrect, I say again" retry branch; campaigns enforce the train/held-out split and never debrief held-out sessions.
- **Agent adapters** (`agents/`): `PilotAgent`/`LearningAgent` protocol, `OpenAICompatPilot` (any real OpenAI-compatible chat-completions endpoint), `HermesPilot` and `WebsocketVoicePilot` (contract-complete, documented as untested against a live gateway — see their docstrings).
- **Live Bolna integration** (`orchestrator/live_bolna.py`, `run_live_call.py`): places a real outbound phone call to a real Bolna voice agent, waits for it to finish, pulls the real transcript, matches every spoken line back to the right scenario turn, and grades it with the unmodified scorer. **4 real calls run so far** — see "Live Bolna calls" below.
- **Storage** (`db/`): SQLite, indexed, three tables (`sessions`, `turns`, `skill_snapshots`).
- **API + dashboard** (`api.py`, `dashboard/index.html`): FastAPI + a single-file SPA, no build step, two tabs:
  - **Flight Sim** — a pixel-art radar/cockpit scene that plays back any real session turn by turn: the controller's real line appears as a speech bubble, the pilot's real reply follows, a correct readback pops a coin/score animation, a scored safety violation (like falling for the callsign trap) shakes the plane, flashes the screen, and costs a life, with the actual intruder aircraft flying past for the callsign-trap case. Every line, every score, every violation shown is pulled live from `/api/sessions/{id}` — nothing in this view is scripted for effect.
  - **Metrics** — leaderboard, learning-curve chart, skill-diff viewer, full-transcript session replay with red safety flags, and a real-campaign launcher.
  - Dark-primary radar/HUD visual design: a validated categorical palette (blue/amber for train-vs-held-out, run through a CVD-separation + contrast validator rather than eyeballed), status colors reserved for score severity and never reused for anything else, score meters instead of ad hoc badges, a hero stat row, and a light-mode variant.
  - `GET /api/config` reports which real capabilities are actually configured (`llm_campaigns_available`, `bolna_live_calls_available`) so the UI can say so honestly instead of a button silently doing nothing or falling back to something fake.
- **Hermes learning-loop mechanics** (`loop/`): debrief protocol + leakage lint (banned-phrase list, blocks any prescriptive text before it's sent) + skill-directory snapshotting/diffing. Unit-tested directly against hand-built inputs (`tests/`) — a real *demonstration* (an actual score-vs-session learning curve) needs a real `LearningAgent` run through a campaign, which needs a live Hermes gateway this workspace doesn't have credentials for.
- **Exporters** (`export/`): JSONL trajectory export, and an Atropos-style `reset()/step()` env wrapper (`RogerThatEnv`), tested against literal ground-truth-derived replies (see `tests/test_atropos_env.py`) rather than a simulated agent.
- **Voice channel DSP** (`channel/`): real bandpass + soft-clip + additive-noise-at-target-SNR + squelch-tail chain (`scipy.signal`), tested against a synthetic-tone generator across all 6 SNR tiers (clean/20/15/10/8/5 dB) — this is a signal-processing unit-test fixture, not an agent, and never produced or stood in for evaluation results.

## What's deliberately stubbed, and why

No Hermes gateway, no LLM key, no TTS/ASR credentials, and no Atropos package are available in this workspace. Rather than fake a passing integration:

- `agents/openai_compat.py` is real, general-purpose code (any OpenAI-compatible endpoint) but has no key configured here — `GET /api/config` reports `llm_campaigns_available: false` until one is set, and the dashboard's campaign launcher refuses to start (400, not a silent no-op) without one.
- `agents/hermes.py` implements the exact `PilotAgent`/`LearningAgent` contract the orchestrator needs, against a documented HTTP wire format, but has never talked to a real Hermes gateway. Point `HERMES_GATEWAY_URL` at one to find out if the wire format needs adjusting.
- `channel/tts.py` / `channel/asr.py` define the pluggable interfaces; `SyntheticTTS` (tone bursts, no credentials) is what the DSP tests run against — a signal-processing fixture, not an agent. `NullASR` raises loudly instead of silently returning empty text.
- `export/atropos_env.py` follows the standard gym `reset()/step()` shape but hasn't been run inside `atroposlib` itself.
- `GET /api/sessions/{id}/audio/{turn}` returns 501 — there's no live TTS provider to have archived audio from.

Wiring any of these to a real provider should not require touching the scorer, orchestrator, or dashboard — that's the point of the adapter boundary.

## Live Bolna calls (real agent, real phone call, real scorer)

This harness has been run against a genuinely live target: a real Bolna voice agent ("ATC Pilot", running `gpt-4.1-mini`) taking a real outbound phone call, with a human reading the controller's lines live. `orchestrator/live_bolna.py` automates everything after the human starts talking:

```bash
python run_live_call.py --scenario callsign-trap-01 --seed 42 \
    --agent-id <bolna-agent-id> --to <e164-phone-number>
```

It places the call, polls until it ends, pulls the real transcript, **matches each spoken line back to the correct scenario turn regardless of what order it was actually read in** (a human controller reading out of order is exactly what happened in testing — matching scores full token overlap between the spoken line and each candidate turn's own canonical script line, with the callsign-addressed-to check as a hard filter only when confidently known, falling back to content alone when ASR has mangled the callsign beyond recognition), grades it with the unmodified scorer, and saves it into `roger_that.db` tagged `pool=live`.

**What's still manual**: the controller's voice. The pilot agent is built to wait and let ATC speak first, so a human reads the scripted lines into the call. Fully automating that side means a second Bolna agent placing an agent-to-agent call, which needs a purchased or connected inbound phone number — real recurring cost, deliberately not done without an explicit go-ahead. The dashboard's Metrics tab reflects this honestly: it generates the exact CLI command for you rather than exposing a one-click "place a real phone call" button that could fire by accident.

The agent's system prompt ([`atc-pilot-agent-prompt.txt`](atc-pilot-agent-prompt.txt)) is deliberately minimal — identity, tone, and "respond as a pilot would" — with zero phraseology coaching, since teaching the rules would invalidate the eval it's the subject of.

## History: nine real bugs, and why the mock pilot is gone

Before any live call was placed, `agents/scripted_mock.py` held a `ScriptedMockPilot`/`LearningMockPilot` pair — a rule-based pilot with a configurable error rate, used only to prove the harness's plumbing (scorer, orchestrator, storage, dashboard) was wired correctly before any real target existed to test it against. Once real calls started, testing against them (plus targeted synthetic adversarial inputs at the unit level) found nine real bugs the mock structurally could not have surfaced, because it never produced the phrasing, disfluencies, or rendering quirks that triggered them:

1. **Trailing-digit swallowing**: the normalizer was silently dropping the last digit of any number at the end of a sentence (`"...QNH one zero two one."` extracted as `1021` → `102`, because the sentence-final period fused onto the word "one."). Fixed in `scorer/normalize.py`.
2. **Keyword-gated safety miss**: `grade_hearback_challenge` only flagged a missed correction if the reply used one of five specific "affirmative" keywords. A real model accepted a wrong frequency by saying *"Confirmed, ..."* — not on the list — and scored a false `safety_score: 1.0` with zero penalty. Fixed by making "did not challenge it" the violation condition directly, not gated on which words it used to not-challenge it.
3. **Narrow negative-word detection**: `is_negative` only recognized `{negative, unable, correction, wrong, incorrect}` — the exact ICAO word or near it. Plain English rejections ("no, that's not correct", "nope", contractions like "I don't think that's right") were all invisible, which would have wrongly penalized a pilot who *correctly* caught an error but phrased it naturally. Also fixed a related tokenizer bug where apostrophes were stripped, breaking every contraction. Broadened `NEGATIVE_WORDS` and fixed the tokenizer in `scorer/normalize.py`.
4. **Disfluency breaks callsign matching**: "victor, uh, tango, alpha bravo charlie" fractured into `"v"` + `"tabc"` because the filler word broke the NATO-letter grouping run, and the two fragments no longer sit adjacent in the substring-matched string. Checked first whether this could cause a *missed safety violation* — it can't, `grade_trap` only checks whether the reply is non-empty, not whether it contains a callsign — so this only affected the `callsign_discipline` metric, not trap detection itself. Fixed by making common disfluency words (`uh`, `um`, `er`, ...) transparent to letter/digit-run grouping in `scorer/normalize.py`.
5. **Two real gaps in state-probe / synonym handling**: "altitude" was not recognized as a synonym for "flight level" at all (`"our altitude is eight zero"` extracted nothing), and a bare numeric answer to a direct probe question (`"eight zero"` in response to "confirm your assigned flight level") failed because no keyword was present. Both are natural, common phrasings. Added `altitude` as an `fl` keyword in `scorer/grammar.py`, and added a scoped bare-number fallback in `grade_state_probe` specifically — safe there in a way it wouldn't be in a general multi-item clearance readback, because the probe question itself disambiguates which single item a bare number refers to.
6. **The hearback-error corruption math could silently generate a fault-free "fault"**: a heading near zero (e.g. 10°) offset downward by the corruption delta went negative (-10), and `spell_digits()` takes `abs()` before speaking a number aloud — so the "corrupted" hearback challenge was spoken *identically* to the truth ("one zero" for both -10 and 10). A pilot that caught nothing would have scored a false "caught it," because there was nothing audible to catch. This is worse than a scorer bug: the test itself was broken, not just the grading of it. Fixed with proper wraparound for headings (mod 360, mapped to the conventional 001-360 range) and domain clamping for the other item types, plus a guard that flips the corruption direction if bounds/wraparound ever collapse the fault back onto the truth. Verified with an exhaustive sweep across the full domain of every item type (hdg/qnh/fl/spd/squawk/freq) at multiple seeds each — zero collisions post-fix.
7. **Turn-matching used item-overlap scoring only, which scores every non-clearance turn (state probes, hearback challenges) a flat 0.** On a real call where two other turns' controller-spoken callsigns were badly ASR-mangled and also scored 0, a state-probe line that transcribed perfectly still lost a 3-way tie to whichever turn happened to come first in scenario order — and then, even once correctly matched, was graded against an always-empty `expected_values` dict (state-probe turns never populate it; only clearance turns do), so it scored `False` regardless of what the pilot actually said. Fixed both: matching now scores full token overlap against each turn's own canonical script line, and state-probe grading now resolves ground truth from whichever prior clearance turn actually assigned that field.
8. **A decimal number with stray whitespace after the point** (a real reply rendered *"switching to approach 127. 1."* instead of "127.1") broke the strict digit-adjacent decimal rule and lost the frequency value entirely. Fixed by collapsing "digit, period, whitespace, digit" into a proper decimal before the adjacency rule runs.
9. **A total matching failure silently reported a "perfect" 1.0.** `score_session([])` returns 1.0 by design — every empty category defaults to 1.0, which is correct when a scenario legitimately has zero turns of some type, but is actively dangerous at the whole-session level, where it means "nothing was graded" is indistinguishable from "everything was flawless." Added `NoTurnsMatchedError`, raised before any scorecard is computed or persisted if zero spoken lines matched anything — `run_live_call.py` now fails loudly with the unmatched lines shown, instead of writing a fabricated perfect session into the database.

All nine have regression tests (`tests/test_scorer.py`, `tests/test_scenario_loader.py`, `tests/test_live_bolna.py`), and the ≥99% golden-agreement bar still holds. Once the mock had done its job — proving the plumbing worked and then getting outrun by real data as the thing worth testing against — keeping it around risked exactly the failure mode it was built to avoid: a leaderboard where "real" and "simulated" performance look identical at a glance. It's gone; `agents/openai_compat.py`, `agents/hermes.py`, and the live Bolna path are what's left, and none of them fake a result.

Two known, deliberately unfixed limitations remain: ASR homophone confusion between "to" and "two" (e.g. "one to six decimal nine" for "126.9") isn't corrected, since mapping "to"→"two" outright risks worse false positives elsewhere in ordinary sentences; and a dropped keyword between two adjacent rapid-fire items ("heading one eight zero one two zero" with no "flight level" before the second number) merges into one garbled number — arguably correct behavior for an ICAO-compliance eval, since real phraseology mandates a keyword before every value precisely to avoid this ambiguity, and a human controller would face the identical problem.

## Quickstart

```bash
cd roger-that
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# run the full test suite (scorer golden tests, session/campaign integration, DSP chain)
python -m pytest tests/ -q

# serve the API + dashboard on one port
uvicorn api:app --reload
# open http://localhost:8000 -- Flight Sim tab plays back whatever real sessions are in
# roger_that.db (none until you run one of the two paths below); Metrics tab shows the
# leaderboard, learning curve, and campaign launcher, all reading the same real data.
```

The database starts empty. Two ways to put real data in it:

**A real live phone call** (needs `BOLNA_API_KEY` and a Bolna agent already created — see the "Live Bolna calls" section):

```bash
python run_live_call.py --scenario callsign-trap-01 --seed 42 \
    --agent-id <your-bolna-agent-id> --to <e164-phone-number>
```

**A real LLM campaign** (needs `OPENAI_API_KEY` or `ROGER_THAT_LLM_API_KEY`, any OpenAI-compatible endpoint):

```python
from agents.openai_compat import OpenAICompatPilot
from orchestrator import run_campaign, list_scenarios
from db.store import Store

store = Store()
scenarios = list_scenarios()
train, held_out = scenarios[:-1], scenarios[-1:]

run_campaign(
    store=store, campaign_id="gpt-4o-mini-eval", agent_name="gpt-4o-mini",
    agent_factory=lambda i: OpenAICompatPilot(model="gpt-4o-mini"),
    train_scenarios=train, held_out_scenarios=held_out,
    n_sessions=20, base_seed=42,
)
```
or from the dashboard's Metrics tab campaign launcher directly, once a key is set in the environment `uvicorn` runs in.

## Repository layout

Matches the project brief's Sec 9, with one deliberate deviation: scenario fault turns (`hearback_challenge`, `trap`) are written directly into a scenario's `script:` list as their own turn type rather than a separate `faults:` block that mutates the script at runtime — same effect, no dynamic rewriting for the loader to get wrong. See the docstring at the top of `orchestrator/scenario_loader.py`.

```
roger-that/
├── scenarios/            # YAML scenario packs (6 of the v1 catalog's 10)
├── controller/           # scripted controller state machine + phraseology spelling
├── state/                # aircraft state tracker
├── scorer/               # normalizer, grammar, grader (the crown jewel)
├── agents/               # base protocol, openai_compat, hermes, websocket_voice
├── orchestrator/         # scenario_loader, session runner, campaign runner, live_bolna
├── loop/                 # debrief protocol, leakage lint, skill snapshotting
├── channel/              # radio DSP chain, pluggable TTS/ASR
├── db/                   # sqlite schema + Store
├── export/               # jsonl + atropos env wrapper
├── dashboard/            # single-file SPA (Flight Sim + Metrics tabs)
├── api.py                # FastAPI (API + dashboard, one port)
├── run_live_call.py      # CLI: place a real Bolna call, score it, save it
└── tests/                # golden tests, integration tests
```

## Adding a scenario

Add a YAML file to `scenarios/`; it's picked up automatically by `orchestrator.list_scenarios()`. See any existing file for the `vars:`/`script:` schema, or the docstring in `orchestrator/scenario_loader.py`.

## Adding an agent

Implement `PilotAgent` (`start_session`, `on_transmission`, `end_session` — see `agents/base.py`) and pass an `agent_factory` to `run_campaign`. Implement `LearningAgent` (adds `debrief`, `skill_dir`) if the agent should participate in the debrief/skill-snapshot loop.
