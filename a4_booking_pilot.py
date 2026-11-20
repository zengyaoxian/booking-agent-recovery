"""A4 booking experiment. Run with --help to see the offline and model modes."""

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from functools import partial
from hashlib import sha256
import json
import os
from pathlib import Path
from random import Random
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class Booking:
    request_id: str
    day: str
    time: str
    seats: int


@dataclass(frozen=True)
class Scenario:
    request: str
    expected: Booking


BOOKING = Booking("booking-1", "Friday", "14:00", 6)
TIMEOUT_MESSAGE = "Create timed out; write status unknown."
TOOL_CALL_LIMIT = 3
MODEL_CALL_LIMIT = 4
USER_REQUEST = "周五下午两点，帮我预约一间六人会议室。"
PILOT = Scenario(USER_REQUEST, BOOKING)
EVALUATION_CASES = (
    Scenario("星期一上午九点一刻，帮我留一间能坐八个人的会议室。",
             Booking("eval-01", "Monday", "09:15", 8)),
    Scenario("我们周二要开组会，下午三点半，十二个人，麻烦订个会议室。",
             Booking("eval-02", "Tuesday", "15:30", 12)),
    Scenario("请预订周三 11:05 的三人会议室。", Booking("eval-03", "Wednesday", "11:05", 3)),
    Scenario("Could you reserve a meeting room for five people at 4:45 p.m. on Thursday?",
             Booking("eval-04", "Thursday", "16:45", 5)),
    Scenario("Room booking: Saturday at 10:20 in the morning, capacity ten.",
             Booking("eval-05", "Saturday", "10:20", 10)),
    Scenario("I almost said 1 p.m. Please book a room for seven at 2:10 p.m. on Sunday.",
             Booking("eval-06", "Sunday", "14:10", 7)),
)
EVALUATION_SEEDS = (17, 43)
EVALUATION_PROTOCOL = "a4-booking-holdout-v1"
TOOL_FEEDBACK_PREFIX = "Tool result: "
FAULTS = ("none", "before_write", "after_write", "rate_limit", "invalid_args")
POLICIES = ("retry", "typed", "repair")
DAYS = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}
AGENT_INSTRUCTIONS = """You operate a simulated meeting-room booking system.
Use one JSON object per reply, without Markdown or additional text:
{"action":"create","day":"Monday","time":"09:30","seats":4}
{"action":"lookup"}
{"action":"finish","status":"success"} or {"action":"finish","status":"failure"}
Infer the day, 24-hour time and number of seats from the user's request.
The tools operate on the current request ID, which is managed by the runner.
After each action you receive a tool result in JSON. Finish with success only
when the requested reservation is confirmed. You have at most 3 tool calls."""


class BookingTimeout(Exception):
    pass


class BookingRateLimited(Exception):
    pass


class BookingService:
    def __init__(self, fault, request_id=BOOKING.request_id):
        self.fault = fault
        self.request_id = request_id
        self.records = []
        self.trace = []
        self.create_attempts = 0
        self.arguments_injected = False

    def _log_call(self, tool, request_id):
        if len(self.trace) >= TOOL_CALL_LIMIT:
            raise RuntimeError("Tool-call budget exceeded")
        event = {"tool": tool, "request_id": request_id}
        self.trace.append(event)
        return event

    def create(self, booking):
        event = self._log_call("create", booking.request_id)
        event["arguments"] = asdict(booking)
        self.create_attempts += 1
        first_attempt = self.create_attempts == 1
        if first_attempt and self.fault == "rate_limit":
            event["result"] = "RATE_LIMITED"
            raise BookingRateLimited
        if not (first_attempt and self.fault == "before_write"):
            self.records.append(booking)
        if first_attempt and self.fault in {"before_write", "after_write"}:
            event["result"] = TIMEOUT_MESSAGE
            raise BookingTimeout(TIMEOUT_MESSAGE)
        event["result"] = "created"

    def lookup(self, request_id):
        event = self._log_call("lookup", request_id)
        matches = [record for record in self.records if record.request_id == request_id]
        event["result"] = {"matches": len(matches)}
        return matches


def run_case(fault, policy):
    service = BookingService(fault)
    try:
        service.create(BOOKING)
    except BookingTimeout:
        if policy == "retry" or not service.lookup(BOOKING.request_id):
            service.create(BOOKING)
    return {
        "fault": fault,
        "policy": policy,
        "success": service.records == [BOOKING],
        "duplicates": max(0, len(service.records) - 1),
        "tool_calls": len(service.trace),
        "records": [asdict(record) for record in service.records],
        "trace": service.trace,
    }


def call_endpoint(messages, seed=None):
    """Call a configured chat-completions-compatible endpoint once."""
    url = os.environ["A4_MODEL_URL"]
    headers = {"Content-Type": "application/json"}
    if token := os.environ.get("A4_MODEL_KEY"):
        headers["Authorization"] = f"Bearer {token}"
    body = {"model": os.environ["A4_MODEL_ID"], "messages": messages, "temperature": 0}
    if seed is not None:
        body["seed"] = seed
    request = Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    with urlopen(request, timeout=30) as response:
        payload = json.load(response)
    return payload["choices"][0]["message"]["content"], payload.get("usage")


def demo_model(messages, expected=BOOKING):
    """A scripted response source for testing the runner offline."""
    create = json.dumps({"action": "create", "day": expected.day, "time": expected.time, "seats": expected.seats})
    if len(messages) == 2:
        return create, None
    observation = json.loads(messages[-1]["content"][len(TOOL_FEEDBACK_PREFIX):])
    if observation.get("code") == "TIMEOUT_WRITE_UNKNOWN":
        return '{"action":"lookup"}', None
    if observation.get("code") in ("RATE_LIMITED", "INVALID_ARGUMENTS") or observation.get("matches") == []:
        return create, None
    return '{"action":"finish","status":"success"}', None


def execute_action(service, action):
    if action["action"] == "lookup":
        return {"status": "ok", "matches": [asdict(record) for record in service.lookup(service.request_id)]}
    day, time, seats = action.get("day"), action.get("time"), action.get("seats")
    valid = (
        isinstance(day, str) and day in DAYS
        and isinstance(time, str) and re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", time, re.ASCII)
        and type(seats) is int and seats > 0
    )
    if not valid:
        event = service._log_call("create", service.request_id)
        event.update(arguments=action, result="INVALID_ARGUMENTS")
        return {"status": "error", "code": "INVALID_ARGUMENTS",
                "message": "Expected an English weekday, 24-hour HH:MM time, and positive integer seats."}
    try:
        service.create(Booking(service.request_id, day, time, seats))
    except BookingTimeout as error:
        return {"status": "error", "code": "TIMEOUT_WRITE_UNKNOWN", "message": str(error)}
    except BookingRateLimited:
        return {"status": "error", "code": "RATE_LIMITED",
                "message": "Temporary rate limit; this attempt did not write a reservation."}
    return {"status": "created"}


def dispatched_action(service, action):
    # The simulated tool adapter serializes seats incorrectly on the first create.
    # An exact replay carries the same bad arguments; a new model action does not.
    if (
        service.fault == "invalid_args" and not service.arguments_injected
        and action["action"] == "create" and type(action.get("seats")) is int
    ):
        service.arguments_injected = True
        return {**action, "seats": str(action["seats"])}
    return action


def recover_with_policy(service, sent, observation, policy, step):
    """Execute the automatic actions allowed for this one recovery attempt."""
    code = observation["code"]
    if policy == "repair" and code == "TIMEOUT_WRITE_UNKNOWN":
        if len(service.trace) == TOOL_CALL_LIMIT:
            return observation
        observation = execute_action(service, {"action": "lookup"})
        step["automatic_actions"].append({"action": "lookup", "observation": observation})
        if observation["matches"]:
            return observation
    elif policy != "retry" and not (policy == "repair" and code == "RATE_LIMITED"):
        return observation
    if len(service.trace) < TOOL_CALL_LIMIT:
        observation = execute_action(service, sent)
        step["automatic_actions"].append({"action": "retry", "observation": observation})
    return observation


def run_agent(call_model, fault, policy="typed", scenario=PILOT):
    if fault not in FAULTS or policy not in POLICIES:
        raise ValueError("Unknown fault or policy")
    service = BookingService(fault, scenario.expected.request_id)
    messages = [
        {"role": "system", "content": AGENT_INSTRUCTIONS},
        {"role": "user", "content": scenario.request},
    ]
    model_trace = []
    status = "model_budget_exhausted"
    confirmed = False
    recovery_attempted = False
    for _ in range(MODEL_CALL_LIMIT):
        reply, usage = call_model(messages)
        step = {"reply": reply, "usage": usage}
        model_trace.append(step)
        try:
            action = json.loads(reply)
        except (TypeError, ValueError):
            status = "invalid_model_reply"
            break
        if not isinstance(action, dict) or action.get("action") not in ("create", "lookup", "finish"):
            status = "invalid_model_reply"
            break
        if action["action"] == "finish":
            status = action.get("status")
            if status not in ("success", "failure"):
                status = "invalid_model_reply"
            break
        if len(service.trace) >= TOOL_CALL_LIMIT:
            status = "tool_budget_exhausted"
            break
        sent = dispatched_action(service, action)
        observation = execute_action(service, sent)
        step["observation"] = observation
        automatic_recovery = policy == "retry" or (
            policy == "repair" and observation.get("code") in ("TIMEOUT_WRITE_UNKNOWN", "RATE_LIMITED")
        )
        if action["action"] == "create" and observation["status"] == "error" and automatic_recovery:
            if not recovery_attempted:
                recovery_attempted = True
                step["automatic_actions"] = []
                observation = recover_with_policy(service, sent, observation, policy, step)
            if policy == "retry" and observation["status"] == "error":
                status = "retry_exhausted"
                break
        confirmed = confirmed or observation["status"] == "created" or bool(observation.get("matches"))
        feedback = TOOL_FEEDBACK_PREFIX + json.dumps(observation)
        step["feedback"] = feedback
        messages.extend([
            {"role": "assistant", "content": reply},
            {"role": "user", "content": feedback},
        ])
    usage = [step["usage"] for step in model_trace]
    tokens = {
        key: sum(item[key] for item in usage)
        if usage and all(isinstance(item, dict) and type(item.get(key)) is int for item in usage)
        else None
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    return {
        "request": scenario.request,
        "expected": asdict(scenario.expected),
        "fault": fault,
        "policy": policy,
        "status": status,
        "confirmed": confirmed,
        "success": status == "success" and confirmed and service.records == [scenario.expected],
        "wrong_writes": sum(record != scenario.expected for record in service.records),
        "duplicates": max(0, len(service.records) - 1),
        "model_calls": len(model_trace),
        "tool_calls": len(service.trace),
        "tokens": tokens,
        "records": [asdict(record) for record in service.records],
        "model_trace": model_trace,
        "tool_trace": service.trace,
    }


def compare(call_model, model_id, trace_file):
    output = open(trace_file, "x", encoding="utf-8") if trace_file else nullcontext()
    with output as trace:
        print(f"Development comparison | {model_id} | per run: {TOOL_CALL_LIMIT} tool / {MODEL_CALL_LIMIT} model calls")
        print(f"{'Fault':<15} {'Policy':<8} {'Success':<8} {'Dup.':>4} {'Tools':>5} {'Model':>5} {'Tokens':>7}",
              flush=True)
        for fault in FAULTS:
            for policy in POLICIES:
                row = run_agent(call_model, fault, policy)
                if trace:
                    trace.write(json.dumps({**row, "model": model_id}, ensure_ascii=False) + "\n")
                    trace.flush()
                tokens = row["tokens"]["total_tokens"]
                print(f"{fault:<15} {policy:<8} {str(row['success']):<8} {row['duplicates']:>4} "
                      f"{row['tool_calls']:>5} {row['model_calls']:>5} "
                      f"{tokens if tokens is not None else '-':>7}", flush=True)


def evaluation_plan():
    runs = []
    for seed in EVALUATION_SEEDS:
        rng = Random(seed)
        scenarios = [(case, fault) for case in EVALUATION_CASES for fault in FAULTS]
        rng.shuffle(scenarios)
        for case, fault in scenarios:
            policies = list(POLICIES)
            rng.shuffle(policies)
            runs.extend((seed, case, fault, policy) for policy in policies)
    return runs


def total_evaluation_tokens(rows):
    totals = [row["tokens"]["total_tokens"] for row in rows]
    return sum(totals) if all(type(total) is int for total in totals) else None


def print_evaluation_summary(rows):
    print("\nSuccesses by preassigned condition:")
    print(f"{'Condition':<16} " + " ".join(f"{policy:>9}" for policy in POLICIES))
    groups = [(fault, [row for row in rows if row["fault"] == fault]) for fault in FAULTS]
    fault_rows = [row for row in rows if row["fault"] != "none"]
    groups.append(("all_faults", fault_rows))
    for seed in EVALUATION_SEEDS:
        groups.append((f"faults_seed_{seed}", [row for row in fault_rows if row["seed"] == seed]))
    for name, group in groups:
        scores = []
        for policy in POLICIES:
            runs = [row for row in group if row["policy"] == policy]
            scores.append(f"{sum(row['success'] for row in runs)}/{len(runs)}")
        print(f"{name:<16} " + " ".join(f"{score:>9}" for score in scores))

    print("\nOn fault cases only (token cost includes failed runs):")
    print(f"{'Policy':<8} {'Duplicates':>10} {'Wrong writes':>12} {'Tools':>7} "
          f"{'Model':>7} {'Tokens':>9} {'Tokens/success':>14}")
    for policy in POLICIES:
        runs = [row for row in fault_rows if row["policy"] == policy]
        successes = sum(row["success"] for row in runs)
        duplicates, wrong_writes, tool_calls, model_calls = (
            sum(row[key] for row in runs) for key in ("duplicates", "wrong_writes", "tool_calls", "model_calls")
        )
        tokens = total_evaluation_tokens(runs)
        cost = f"{tokens / successes:.1f}" if tokens is not None and successes else "-"
        print(f"{policy:<8} {duplicates:>10} {wrong_writes:>12} {tool_calls:>7} "
              f"{model_calls:>7} {tokens if tokens is not None else '-':>9} {cost:>14}")


def evaluate(use_endpoint, model_id, trace_path):
    plan = evaluation_plan()
    expected_keys = {(seed, case.expected.request_id, fault, policy) for seed, case, fault, policy in plan}
    protocol = {
        "kind": "protocol", "id": EVALUATION_PROTOCOL, "model": model_id,
        "mode": "live" if use_endpoint else "scripted",
        "endpoint_hash": sha256(os.environ["A4_MODEL_URL"].encode()).hexdigest() if use_endpoint else None,
        "seeds": list(EVALUATION_SEEDS), "temperature": 0,
        "faults": list(FAULTS), "policies": list(POLICIES),
        "tool_limit": TOOL_CALL_LIMIT, "model_limit": MODEL_CALL_LIMIT,
        "system_prompt": AGENT_INSTRUCTIONS,
        "cases": [asdict(case) for case in EVALUATION_CASES],
    }
    completed = {}
    path = Path(trace_path)
    if path.exists():
        with path.open(encoding="utf-8") as trace:
            if json.loads(trace.readline()) != protocol:
                raise ValueError("Trace belongs to a different protocol or model; choose a new file.")
            for line in trace:
                row = json.loads(line)
                key = tuple(row[field] for field in ("seed", "case_id", "fault", "policy"))
                if key not in expected_keys or key in completed:
                    raise ValueError("Trace contains an unexpected or duplicate run.")
                completed[key] = row
    else:
        with path.open("x", encoding="utf-8") as trace:
            trace.write(json.dumps(protocol, ensure_ascii=False) + "\n")

    print(f"Held-out evaluation | {model_id} | {len(completed)}/{len(plan)} runs already saved", flush=True)
    with path.open("a", encoding="utf-8") as trace:
        for seed, case, fault, policy in plan:
            key = (seed, case.expected.request_id, fault, policy)
            if key in completed:
                continue
            call_model = (
                partial(call_endpoint, seed=seed) if use_endpoint
                else partial(demo_model, expected=case.expected)
            )
            print(f"[{len(completed) + 1}/{len(plan)}] {key[1]} seed={seed} {fault} {policy}...",
                  end=" ", flush=True)
            try:
                row = run_agent(call_model, fault, policy, case)
            except (URLError, TimeoutError) as error:
                status = f"HTTP {error.code}" if isinstance(error, HTTPError) else type(error).__name__
                raise RuntimeError(f"{status}. {len(completed)} completed runs saved. "
                                   "Repeat the same command to resume.") from None
            row.update(kind="run", seed=seed, case_id=key[1], model=model_id)
            trace.write(json.dumps(row, ensure_ascii=False) + "\n")
            trace.flush()
            completed[key] = row
            print("OK" if row["success"] else f"FAIL ({row['status']})", flush=True)
    print_evaluation_summary(list(completed.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Include replay traces")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--demo-agent", action="store_true", help="Run a scripted offline dialogue")
    mode.add_argument("--agent", action="store_true",
                      help="Use A4_MODEL_URL (full endpoint URL), A4_MODEL_ID, and optional A4_MODEL_KEY")
    parser.add_argument("--fault", choices=FAULTS, default="none",
                        help="Inject a fault in agent mode")
    experiment = parser.add_mutually_exclusive_group()
    experiment.add_argument("--compare", action="store_true", help="Compare three policies on the development cases")
    experiment.add_argument("--evaluate", action="store_true", help="Run the fixed held-out benchmark")
    parser.add_argument("--trace-file", help="Write JSONL traces; --evaluate resumes a matching file")
    args = parser.parse_args()
    if args.agent and not all(os.environ.get(key) for key in ("A4_MODEL_URL", "A4_MODEL_ID")):
        parser.error("--agent requires A4_MODEL_URL and A4_MODEL_ID; omit --agent to run offline")
    if args.trace_file and not (args.compare or args.evaluate):
        parser.error("--trace-file requires --compare or --evaluate")
    if args.evaluate and not args.trace_file:
        parser.error("--evaluate requires --trace-file")
    call_model = call_endpoint if args.agent else demo_model
    model_id = os.environ["A4_MODEL_ID"] if args.agent else "scripted demo"
    if args.evaluate:
        try:
            evaluate(args.agent, model_id, args.trace_file)
        except KeyboardInterrupt:
            parser.exit(130, "\nStopped. Repeat the same command to resume completed work.\n")
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            parser.exit(1, f"\nEvaluation stopped: {error}\n")
        return
    if args.compare:
        try:
            compare(call_model, model_id, args.trace_file)
        except FileExistsError:
            parser.error("--trace-file already exists; choose a new filename")
        return
    if args.demo_agent or args.agent:
        result = run_agent(call_model, args.fault)
        result["model"] = model_id
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    rows = [
        run_case(fault, policy)
        for fault in ("before_write", "after_write")
        for policy in ("retry", "lookup_first")
    ]
    if args.json:
        print(json.dumps(rows, indent=2))
        return
    print(f"{'Timeout':<15} {'Policy':<15} {'Success':<9} {'Duplicates':>10} {'Calls':>6}")
    for row in rows:
        print(
            f"{row['fault']:<15} {row['policy']:<15} {str(row['success']):<9} "
            f"{row['duplicates']:>10} {row['tool_calls']:>6}"
        )


if __name__ == "__main__":
    main()
