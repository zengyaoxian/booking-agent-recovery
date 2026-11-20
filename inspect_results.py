"""Verify the recorded tables or print a tool trace without calling a model."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re

from a4_booking_pilot import Booking, FAULTS, POLICIES, evaluation_plan

ROOT = Path(__file__).resolve().parent / "results"


def checked(condition, message):
    if not condition:
        raise ValueError(message)


def inspect_holdout():
    terminal = (ROOT / "heldout_terminal.txt").read_text(encoding="utf-8")
    patterns = r"^\[(\d+)/180\] (eval-\d+) seed=(\d+) (\w+) (\w+)\.\.\. (OK|FAIL \([^)]+\))$"
    rows = re.findall(patterns, terminal, re.MULTILINE)
    order = [(str(i), case.expected.request_id, str(seed), fault, policy)
             for i, (seed, case, fault, policy) in enumerate(evaluation_plan(), 1)]
    checked([row[:5] for row in rows] == order, "The recorded trials differ from the evaluation plan.")
    by = Counter((fault, policy, outcome == "OK") for _, _, _, fault, policy, outcome in rows)
    print("Held-out results | success counts verified from all 180 terminal rows")
    print(f"{'Condition':<16}" + "".join(f"{p:>10}" for p in POLICIES))
    for fault in FAULTS:
        fields = [f"{by[fault, p, True]}/{by[fault, p, True] + by[fault, p, False]}" for p in POLICIES]
        print(f"{fault:<16}" + "".join(f"{field:>10}" for field in fields))
    print("\nFault runs | call and side-effect totals taken from the terminal summary")
    print("Policy     Success  Dup  Wrong  Tools  Model  Tokens  Tokens/success")
    costs = {}
    for policy in POLICIES:
        match = re.search(
            rf"^{policy}\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+\.\d+)\s*$",
            terminal, re.MULTILINE,
        )
        checked(match is not None, f"Missing summary for {policy}")
        duplicates, wrong, tools, model_calls, tokens = map(int, match.groups()[:5])
        successes = sum(by[fault, policy, True] for fault in FAULTS if fault != "none")
        costs[policy] = tokens / successes
        checked(f"{costs[policy]:.1f}" == match.group(6), f"Cost mismatch for {policy}")
        print(f"{policy:<10} {successes:>2}/48 {duplicates:>5} {wrong:>6} {tools:>6} "
              f"{model_calls:>6} {tokens:>7,} {costs[policy]:>15.1f}")
    print(f"Bounded repair versus typed feedback: {1 - costs['repair']/costs['typed']:.1%} fewer tokens per success.")


def read_development():
    with (ROOT / "development_traces.jsonl").open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source]
    checked(len(rows) == 15 and {(r["fault"], r["policy"]) for r in rows}
            == {(fault, policy) for fault in FAULTS for policy in POLICIES}, "Incomplete development matrix.")
    expected = Booking("booking-1", "Friday", "14:00", 6)
    for row in rows:
        records = [Booking(**record) for record in row["records"]]
        answer = row["status"] == "success" and row["confirmed"] and records == [expected]
        checked(answer == row["success"], "Development score mismatch.")
        checked(max(0, len(records) - 1) == row["duplicates"], "Duplicate count mismatch.")
        checked(row["tokens"]["total_tokens"] == sum(
            step["usage"]["total_tokens"] for step in row["model_trace"]), "Token count mismatch.")
        checked(len(row["model_trace"]) == row["model_calls"], "Model call count mismatch.")
        checked(len(row["tool_trace"]) == row["tool_calls"], "Tool call count mismatch.")
    return rows


def replay(rows, fault):
    by_policy = {row["policy"]: row for row in rows if row["fault"] == fault}
    print(f"Recorded development trace | {fault} | {by_policy['typed']['request']}")
    for policy in POLICIES:
        row = by_policy[policy]
        print(f"\n{policy}: {'PASS' if row['success'] else 'FAIL'}, "
              f"{len(row['records'])} final record(s), {row['model_calls']} model calls, "
              f"{row['tokens']['total_tokens']} tokens")
        for step in row["model_trace"]:
            reply = json.loads(step["reply"])
            print("  Model:", json.dumps(reply, ensure_ascii=False))
            if "observation" in step:
                print("    Tool response:", json.dumps(step["observation"], ensure_ascii=False))
            for operation in step.get("automatic_actions", ()):
                print("    Program:", operation["action"],
                      json.dumps(operation["observation"], ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", choices=FAULTS, metavar="FAULT", help="Print a recorded development case")
    args = parser.parse_args()
    rows = read_development()
    if args.replay:
        replay(rows, args.replay)
    else:
        inspect_holdout()
        print("All 15 development traces separately checked.")


if __name__ == "__main__":
    main()
