"""Skill archaeology: snapshot an agent's skill directory (agentskills.io
-format files) after every session and store the raw content so the
dashboard's skill-diff viewer can render how the agent's self-written
atc-phraseology skill evolved. The harness only ever reads this directory --
see the injection boundary note in agents/hermes.py.
"""
from __future__ import annotations

import difflib
from pathlib import Path
from typing import Dict, List


def read_skill_dir(skill_dir: str) -> Dict[str, str]:
    root = Path(skill_dir)
    if not root.exists():
        return {}
    return {p.name: p.read_text() for p in sorted(root.glob("*")) if p.is_file()}


def diff_content(old: str, new: str) -> List[str]:
    return list(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile="before",
            tofile="after",
        )
    )
