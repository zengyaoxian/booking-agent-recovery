# Bounded Recovery for a Meeting Room Booking Agent

CSC5051 Natural Language Processing · Track A, Brief A4 · Fall 2026

## Abstract

A booking call can time out after the reservation has already been recorded. Retrying in that situation can create a second reservation. We compare plain retry, model-led recovery with typed errors, and bounded rule-based recovery under equal call limits. We ran 180 trials with Qwen Plus in a simulator. Over 48 fault trials per policy, typed feedback and bounded repair both completed 48 tasks with zero duplicates. Plain retry completed 24 tasks and made 12 duplicate writes. Bounded repair used 502.2 tokens per successful recovery, 38.2% fewer than typed feedback, and made 49 fewer model calls. The study tests one hosted model and a simulator with immediate, reliable lookups.

## 1. Introduction and Related Work

The same timeout can hide two outcomes. If a reservation was never written, the agent should make another attempt. If the reservation was written, another attempt can create a duplicate. An agent can check the reservation store or delegate that check to the program. We ask whether moving routine recovery into the program reduces token use while maintaining task success under the same limits.

Anthropic distinguishes workflows with predefined code paths from agents in which a language model chooses its next action [1]. We compare these choices within the recovery stage of a single agent. We follow Kapoor et al. in evaluating task success together with its cost [2], and adapt the goal-state evaluation used in τ-bench [3]. Our setup isolates four injected faults in a small booking service so that the outcomes of different responses to the same fault can be compared directly.

## 2. Agent and Recovery Policies

The agent reads a natural-language request and produces a JSON (JavaScript Object Notation) action. It can create a booking with a weekday, time, and seat count, look up bookings by the current request identifier, or declare the task finished. A standard-library Python loop sends messages to the same hosted model, executes tool actions, and returns observations. The model sees the request and tool observations. The scorer separately reads the simulator's final records and the expected booking.

**Plain retry.** After a failed create, the program issues the same dispatched call once. If this retry fails, the run ends. The program does not look up the booking before retrying.

**Typed feedback.** The program sends an error code and message to the model. The model chooses the next action, such as looking up the booking or issuing a corrected create.

**Bounded repair.** After an ambiguous timeout, the program looks up the request identifier and resends the create only when no booking is found and the budget permits. After an explicit rate limit, it retries without a lookup because that attempt did not write. Invalid-argument errors are handled by the model as in typed feedback. The program starts automatic recovery at most once per run. It returns the final observation to the model for the next decision.


## 3. Evaluation

We froze six booking requests, three Chinese and three English, separate from the development request. Each crossed five conditions, three policies, and two request seeds (17 and 43), for 180 runs. The seeds were sent to the hosted model, and seeded randomization changed execution order. All runs used the `qwen-plus` API (Application Programming Interface) model alias, temperature 0, a shared system prompt, and limits of three tool calls and four model calls. Each run started with an empty booking store.

The four faults were a timeout just before a write, a timeout just after a write, a temporary rate limit before any write, and an invalid seat-count argument introduced by converting an integer to a string on the first eligible create. Both timeout conditions return the same error. An exact replay of an invalid call preserves the bad argument, while a fresh action from the model can correct it. A no-fault condition provides a control.

A run scores as successful only when the model declares success, a tool observation confirms a booking, and the store contains exactly one record that matches the request. We count additional records for the same request as duplicate writes, and count records with fields differing from the expected booking as wrong writes. Tokens per success divides all input and output tokens spent on fault runs, including failed runs, by the number of successful fault runs. We report these counts descriptively.

## 4. Results

Table 1. Successful runs by condition.

| Condition | Plain retry | Typed feedback | Bounded repair |
|---|---:|---:|---:|
| No fault | 12/12 | 12/12 | 12/12 |
| Timeout before write | 12/12 | 12/12 | 12/12 |
| Timeout after write | 0/12 | 12/12 | 12/12 |
| Rate limit | 12/12 | 12/12 | 12/12 |
| Invalid arguments | 0/12 | 12/12 | 12/12 |
| All fault runs | 24/48 | 48/48 | 48/48 |

Table 2. Side effects and cost on fault runs only.

| Metric | Plain retry | Typed feedback | Bounded repair |
|---|---:|---:|---:|
| Duplicate writes | 12 | 0 | 0 |
| Wrong writes | 0 | 0 | 0 |
| Tool calls | 96 | 109 | 108 |
| Model calls | 84 | 157 | 108 |
| Total tokens | 16,996 | 39,005 | 24,108 |
| Tokens per success | 708.2 | 812.6 | 502.2 |

At each seed, the three policies recovered 12/24, 24/24, and 24/24 of the fault runs, respectively. Bounded repair and typed feedback used almost the same number of tool calls. Moving routine tool decisions into the program reduced model calls by 31.2% and tokens per success by 38.2%.


## 5. Failure Analysis and Limitations


**Held-out failure.** On post-write timeout case `eval-01` at seed 17, the independent scorer rejected plain retry even though the model reported success. Both other policies passed. Plain retry also exhausted its retry on all 12 invalid-argument cases. Both policies that let the model revise arguments passed all 12.

**Recorded mechanism.** Table 3 shows the corresponding post-write timeout from the separate development run. Blind retry left two records. Both successful policies conducted the same lookup, but bounded repair performed it within the program and saved one model call.

Table 3. Recorded post-write timeout in the development trace.

| Policy | Recovery action after timeout | Final records | Model calls |
|---|---|---:|---:|
| Plain retry | Program resends create | 2 | 2 |
| Typed feedback | Model requests lookup | 1 | 3 |
| Bounded repair | Program runs lookup | 1 | 2 |

**Scope.** We used one model alias and six requests repeated across two seeds. A lookup sees an isolated store immediately, so an empty lookup is decisive here. This assumption would need verification with delayed writes or concurrent requests. The balanced faults do not estimate real-world frequencies. We did not measure latency or monetary cost.

## Reproducibility and Contribution

The simulator and evaluation runner are contained in `a4_booking_pilot.py`. The script freezes requests and seeds and saves a JSON Lines trace. The accompanying package contains the development traces, the held-out terminal summary, and scripts to verify the tables and print the recorded example. With `A4_MODEL_URL`, `A4_MODEL_ID`, and a valid `A4_MODEL_KEY` set in the environment, the live evaluation command is:

```sh
python3 a4_booking_pilot.py --agent --evaluate --trace-file a4_booking_eval.jsonl
```

The student author selected the project, configured the hosted model connection, ran the evaluation, and inspected the results and recovery policies.

## References

[1] Anthropic. 2024. Building Effective Agents. https://www.anthropic.com/research/building-effective-agents

[2] Sayash Kapoor, Benedikt Stroebl, Zachary S. Siegel, Nitya Nadgir, and Arvind Narayanan. 2025. AI Agents That Matter. Transactions on Machine Learning Research. https://openreview.net/forum?id=Zy4uFzMviZ

[3] Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan. 2025. τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains. International Conference on Learning Representations. https://arxiv.org/abs/2406.12045
