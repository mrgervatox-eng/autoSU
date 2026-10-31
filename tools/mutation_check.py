"""Mutation check: break the safety code on purpose and see whether the tests notice.

For every comparison, arithmetic operator, and/or/not, True/False, max/min and any/all in the target
files, make ONE change in a throwaway copy of the project and run the fast test suite against it.
  killed    = some test failed (good: the tests would catch that bug)
  survived  = every test still passed (a gap, or an equivalent change that cannot matter)

  python tools/mutation_check.py                 # validator, constraints, engine, fallback
  python tools/mutation_check.py --file negotiator/log.py
  python tools/mutation_check.py --jobs 4 --limit 20
"""
from __future__ import annotations

import argparse
import io
import os
import shutil
import subprocess
import sys
import tempfile
import tokenize
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARGETS = ["negotiator/validator.py", "negotiator/constraints.py", "negotiator/engine.py", "negotiator/fallback.py",
                   "negotiator/summary.py"]
FAST_TESTS = ["tests/test_validator.py", "tests/test_validator_verification.py", "tests/test_engine.py",
              "tests/test_fallback_log.py", "tests/test_schemas_config.py", "tests/test_summary.py"]

OPS = {">=": ">", "<=": "<", ">": ">=", "<": "<=", "==": "!=", "!=": "==", "+": "-", "-": "+", "*": "+"}
NAMES = {"and": "or", "or": "and", "True": "False", "False": "True", "max": "min", "min": "max", "any": "all", "all": "any"}


@dataclass
class Mutant:
    file: str
    line: int
    col: int
    end_col: int
    original: str
    replacement: str
    text: str

    @property
    def label(self) -> str:
        return f"{self.file}:{self.line}  {self.original!r} -> {self.replacement!r}   | {self.text}"


def mutants_for(rel: str) -> list[Mutant]:
    src = (ROOT / rel).read_text(encoding="utf-8")
    lines = src.splitlines()
    toks = [t for t in tokenize.generate_tokens(io.StringIO(src).readline) if t.type in (tokenize.OP, tokenize.NAME)]
    out: list[Mutant] = []

    def add(t_start, t_end, orig, repl):
        (l1, c1), (l2, c2) = t_start, t_end
        if l1 == l2:
            out.append(Mutant(rel, l1, c1, c2, orig, repl, lines[l1 - 1].strip()[:90]))

    for i, t in enumerate(toks):
        line = lines[t.start[0] - 1]
        if line.lstrip().startswith(("import ", "from ")):
            continue
        if t.type == tokenize.OP and t.string in OPS:
            add(t.start, t.end, t.string, OPS[t.string])
        elif t.type == tokenize.NAME:
            if t.string == "not" and i + 1 < len(toks) and toks[i + 1].string == "in":
                add(t.start, toks[i + 1].end, "not in", "in")
            elif t.string == "in" and not (i > 0 and toks[i - 1].string == "not") and " for " not in line and not line.lstrip().startswith("for "):
                add(t.start, t.end, "in", "not in")
            elif t.string in NAMES and not (t.string in ("max", "min", "any", "all") and (i + 1 >= len(toks) or toks[i + 1].string != "(")):
                add(t.start, t.end, t.string, NAMES[t.string])
    return out


def apply(m: Mutant, text: str) -> str:
    lines = text.splitlines(keepends=True)
    row = lines[m.line - 1]
    lines[m.line - 1] = row[: m.col] + m.replacement + row[m.end_col :]
    return "".join(lines)


def make_workspace(base: Path) -> Path:
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "logs")
    for d in ("negotiator", "tests", "docs"):
        shutil.copytree(ROOT / d, base / d, ignore=ignore)
    shutil.copy(ROOT / "config.example.toml", base / "config.example.toml")
    return base


def run_tests(ws: Path) -> int:
    env = dict(os.environ, FUZZ_SCALE=os.environ.get("FUZZ_SCALE", "0.25"), PYTHONDONTWRITEBYTECODE="1")
    env.pop("NEGOTIATOR_CONFIG", None)
    return subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", *FAST_TESTS],
                          cwd=ws, env=env, capture_output=True, timeout=300).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", action="append", help="file(s) to mutate, relative to the project root")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="only try the first N mutants (smoke test)")
    args = ap.parse_args()

    muts = [m for rel in (args.file or DEFAULT_TARGETS) for m in mutants_for(rel)]
    if args.limit:
        muts = muts[: args.limit]
    tmp = Path(tempfile.mkdtemp(prefix="mutcheck_"))
    try:
        spaces = [make_workspace(tmp / f"w{i}") for i in range(args.jobs)]
        if run_tests(spaces[0]) != 0:
            print("The unmodified project fails its own tests; fix that first.")
            return 2
        print(f"{len(muts)} mutants, {args.jobs} workers; baseline passes.\n")

        def trial(idx_m):
            idx, m = idx_m
            ws = spaces[idx % args.jobs]
            path = ws / m.file
            original = path.read_text(encoding="utf-8")
            try:
                path.write_text(apply(m, original), encoding="utf-8")
                code = run_tests(ws)
            except subprocess.TimeoutExpired:
                code = 124  # a hang counts as noticed
            finally:
                path.write_text(original, encoding="utf-8")
            return m, code

        # one worker per workspace at a time: group mutants by workspace index
        results: list[tuple[Mutant, int]] = []
        buckets = [[(i, m) for i, m in enumerate(muts) if i % args.jobs == w] for w in range(args.jobs)]
        with ThreadPoolExecutor(args.jobs) as pool:
            for chunk in pool.map(lambda b: [trial(x) for x in b], buckets):
                results.extend(chunk)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    survivors = [m for m, code in results if code == 0]
    killed = len(results) - len(survivors)
    print(f"killed {killed} of {len(results)} mutants ({100 * killed / max(1, len(results)):.0f}%)")
    if survivors:
        print(f"\n{len(survivors)} survived:")
        for m in sorted(survivors, key=lambda m: (m.file, m.line, m.col)):
            print("  " + m.label)
    return 0 if not survivors else 1


if __name__ == "__main__":
    raise SystemExit(main())
