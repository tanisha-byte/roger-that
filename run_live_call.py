#!/usr/bin/env python3
"""Runs one Roger That scenario as a real Bolna phone call, fully automated
from the moment you pick up: places the call, prints the script sheet for
you to read, waits for the call to finish, pulls the real transcript,
matches lines back to scenario turns (any order), grades with the real
scorer, and saves the session into roger_that.db.

Usage:
  python run_live_call.py --scenario callsign-trap-01 --seed 42 \
      --agent-id <your-bolna-agent-id> --to <e164-phone-number>

The controller side is still your voice -- see the printed lines below.
Automating that too means a second Bolna agent placing an agent-to-agent
call, which needs a purchased/connected inbound number (real recurring
cost); see orchestrator/live_bolna.py's module docstring.
"""
from __future__ import annotations

import argparse
import json
import sys

from db.store import Store
from orchestrator.live_bolna import NoTurnsMatchedError, run_live_bolna_session
from orchestrator.scenario_loader import load_scenario_yaml, render_scenario, SCENARIOS_DIR


def print_script_sheet(scenario: dict, seed: int) -> None:
    rs = render_scenario(scenario, seed)
    print(f"\nCALLSIGN: {rs.callsign['display']}")
    intruder = rs.ctx.get("intruder")
    if intruder:
        print(f"INTRUDER (reference only -- do not read as your own callsign): {intruder['display']}")
    print("\nWait for the agent to say its opening line, then read these in order:\n")
    for t in rs.turns:
        note = ""
        if t.type == "clearance":
            note = f"  (expects readback of: {[e['item'] for e in t.expect_readback]})"
        elif t.type in ("trap", "clutter"):
            note = "  (the trap -- addressed to the intruder. Correct behavior: stay silent.)"
        print(f"  Turn {t.index} [{t.type}]: {t.atc_text}{note}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="callsign-trap-01")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--agent-id", required=True)
    ap.add_argument("--to", required=True, help="recipient phone number, E.164")
    ap.add_argument("--agent-name", default="bolna-live-gpt4.1-mini")
    ap.add_argument("--campaign-id", default="live-bolna-call-2026-07-03")
    ap.add_argument("--timeout", type=float, default=180.0)
    args = ap.parse_args()

    scenario = load_scenario_yaml(SCENARIOS_DIR / f"{args.scenario}.yaml")
    print_script_sheet(scenario, args.seed)

    input("Press enter to place the call now (this rings a real phone)... ")

    store = Store()
    try:
        result = run_live_bolna_session(
            store=store, scenario=scenario, seed=args.seed, agent_id=args.agent_id,
            recipient_phone_number=args.to, agent_name=args.agent_name,
            campaign_id=args.campaign_id, poll_timeout_s=args.timeout,
        )
    except NoTurnsMatchedError as e:
        print(f"\nFAILED -- no scorecard produced, nothing saved: {e}")
        store.close()
        sys.exit(1)

    print(f"\nexecution_id: {result.execution_id}")
    print(f"\nraw transcript:\n{result.transcript}")
    if result.unmatched_lines:
        print(f"\nWARNING -- {len(result.unmatched_lines)} spoken line(s) could not be matched to a scenario turn "
              f"and were not scored:")
        for line in result.unmatched_lines:
            print(f"  - {line}")
    print(f"\nscorecard:\n{json.dumps(result.scorecard, indent=2)}")
    store.close()


if __name__ == "__main__":
    sys.path.insert(0, ".")
    main()
