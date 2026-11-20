# Booking Agent Recovery

A meeting-room booking call can time out after the reservation has already been written. This project compares three ways for an agent to recover under the same call budget using a simulated booking service.

**Bounded repair recovered all 48 fault trials without duplicates and used 38.2% fewer tokens per successful recovery than model-led recovery with structured errors.**

## Replay the experiment

Use Python 3.8 or newer. The project uses the Python standard library and was verified with Python 3.12.14. These commands run offline.

```sh
python3 inspect_results.py
python3 inspect_results.py --replay after_write
```

The first command checks all 180 held-out outcome rows against the fixed evaluation plan, recomputes success counts and tokens per success, and verifies the 15 development traces. The second replays a recorded development case. After a post-write timeout, plain retry sends another create call and leaves two bookings. Both other policies look up the existing booking. Bounded repair runs the lookup in the program, saving one model call in that replay.

## Three recovery policies

| Policy | Action after a failed create |
| --- | --- |
| Plain retry | The program resends the same call once. |
| Structured errors | The model sees the error code and chooses the next action. |
| Bounded repair | The program checks the booking store after an ambiguous timeout and retries only when the lookup is empty and budget remains. It retries an explicit pre-write rate limit and lets the model fix invalid arguments. |

## Results

The evaluation crossed six held-out requests, five conditions, three policies, and two seeds for a total of 180 trials. Each trial allowed up to three tool calls and four model calls. Success required a model success declaration, tool confirmation, and exactly one correct booking in the simulator.

| Metric on fault trials | Plain retry | Structured errors | Bounded repair |
| --- | ---: | ---: | ---: |
| Successful recoveries | 24/48 | 48/48 | 48/48 |
| Duplicate writes | 12 | 0 | 0 |
| Tool calls | 96 | 109 | 108 |
| Model calls | 84 | 157 | 108 |
| Tokens per successful recovery | 708.2 | 812.6 | 502.2 |

All three policies passed their 12 no-fault control trials. The experiment used one hosted model and a simulator with immediate, reliable lookups. Delayed writes, concurrent requests, and end-to-end latency were outside the evaluation. The full per-fault results and study design are in the [project report](docs/report.md).

## Run the evaluation with a hosted model

The original experiment used Alibaba Cloud Model Studio's `qwen-plus` alias at temperature 0 with seeds 17 and 43. The saved results were archived on 2026-11-20. The original logs omit the exact run timestamp and underlying model snapshot, so a fresh run can produce different results.

In Bash or Zsh, enter the full chat-completions endpoint for your region when prompted, then paste your key at the next prompt. The key is read without echoing. See the [Model Studio endpoint documentation](https://www.alibabacloud.com/help/en/model-studio/compatibility-of-openai-with-dashscope).

```sh
printf 'Full chat-completions URL: '
read -r A4_MODEL_URL
printf 'Model Studio key: '
read -rs A4_MODEL_KEY
printf '\n'
export A4_MODEL_URL A4_MODEL_KEY
export A4_MODEL_ID=qwen-plus
python3 a4_booking_pilot.py --agent --evaluate --trace-file a4_booking_eval.jsonl
```

Repeat the last command with the same settings and trace path to resume an interrupted run.

## Repository contents

| Path | Contents |
| --- | --- |
| [`a4_booking_pilot.py`](a4_booking_pilot.py) | Simulator, recovery policies, fixed evaluation cases, and evaluation runner |
| [`inspect_results.py`](inspect_results.py) | Offline verification and case replay |
| [`results/heldout_terminal.txt`](results/heldout_terminal.txt) | Original terminal outcomes and aggregate statistics for 180 held-out trials |
| [`results/development_traces.jsonl`](results/development_traces.jsonl) | Step-by-step traces for the separate 15-trial development comparison |
| [`docs/report.md`](docs/report.md) | Methods, complete results, and limitations |
| [`requirements.txt`](requirements.txt) | Python environment requirements |

Course project for CSC5051 Natural Language Processing, Track A, Brief A4, Fall 2026, The Chinese University of Hong Kong, Shenzhen. [Course project guide](https://nlp-course-cuhksz.github.io/final-project.html).
