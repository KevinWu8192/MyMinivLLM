# Tests and Benchmarks

All verification assets live under `tests/`:

```text
tests/
├── test_*.py                  # Pytest correctness and regression tests
├── benchmarks/                # CUDA and end-to-end performance drivers
└── results/qwen3_32b/         # Committed Qwen3-32B benchmark evidence
```

## Unit and Regression Tests

## Setup

```bash
pip install pytest xxhash
```

## Run

Run all Pytest tests:
```bash
uv run pytest -q
```

Run scheduler tests or a specific scheduler regression:
```bash
uv run pytest tests/test_scheduler.py -v
uv run pytest tests/test_scheduler.py::TestBug2TokenLimitBreak -v
uv run pytest tests/test_scheduler.py::TestBug1CanAppendFailure -v
uv run pytest tests/test_scheduler.py::TestSchedulerHappyPath -v
```

### Scheduler Test Classes

### TestBug2TokenLimitBreak

Guards against sequences being silently dropped when the token budget (or sequence-count limit) is exhausted mid-loop.

Tests:
- `test_seq_count_is_correct` — only 2 sequences fit in a 2-token budget; `seq_c` must remain in `running`
- `test_seq_count_limit_variant` — same bug triggered by `max_num_sequences` instead of token budget
- `test_no_sequence_is_lost` — total sequence conservation: every sequence must be in `running`, `waiting`, or `scheduled`

### TestBug1CanAppendFailure

Guards against sequences being lost when `block_manager.can_append` returns `False`.

Tests:
- `test_seq_a_not_lost` — `seq_a` must appear in `running`, `waiting`, or `scheduled` after the call
- `test_total_conservation` — neither `seq_a` nor `seq_b` may disappear

### TestSchedulerHappyPath

Basic correctness of the scheduler under normal conditions.

Tests:
- `test_prefill_scheduled_first` — a newly added sequence is scheduled as prefill and moved to `running`
- `test_all_running_seqs_scheduled_when_budget_allows` — when the token budget is large enough, all running sequences are scheduled and remain in `running`
- `test_preempt_only_seq_when_cant_append_and_running_empty` — when the only running sequence cannot append, it is preempted to `waiting` with status `WAITING`

## Benchmarks and Results

See [benchmarks/README.md](benchmarks/README.md) for the benchmark catalog and
[results/qwen3_32b/README.md](results/qwen3_32b/README.md) for the committed
Qwen3-32B hardware configuration, commands, results, and analysis.
