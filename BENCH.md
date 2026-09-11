# Part A benchmark

Cases: calendar rejects sticky; systems-design accepts it; nutrition rejects it; mixed
calendar/CAP usually accepts it; empty candidates do not crash.

The benchmark is `benchmarks/bench_gate.py`. It measures deterministic local gate
overhead, excluding provider latency and cost:

```powershell
python -m benchmarks.bench_gate --iterations 1000 --output benchmarks/results/fixture.json
```

Generated results go in ignored `benchmarks/results/`. The report includes evaluations,
median and p95 latency, and accepted count. An uncached live candidate makes three judge
calls; use provider usage data for live cost.
