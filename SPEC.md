PROJECT: Multi-agent air traffic disruption negotiator (hackathon, 24h).

BRIEF: ATC, Pilot and Ground agents must negotiate a resolution to a
disruption using structured messages, explicit constraint checking, a
traceable consensus log, and a deterministic non-AI fallback. Forbidden:
combining roles in one prompt, unbounded loops, accepting solutions that
violate safety margins.

ARCHITECTURE:
- Python + FastAPI + WebSockets. Python state machine owns the loop;
  LLMs are only actors inside it.
- 3 separate Claude calls (one system prompt per role), JSON-only output.
- Deterministic validator checks every proposal and every "ACCEPTED"
  against numeric constraints. It can veto an LLM acceptance.
- Max 3 negotiation rounds, then deterministic fallback (conservative
  hold + reassignment). Label it "our conservative default procedure";
  do not cite real regulations.
- Human-in-the-loop: after consensus, ATC agent shows the controller a
  ranked recommendation (2 options). Approve, or reject with free text
  that is parsed into a structured hard constraint and triggers
  renegotiation (max 2 times, plus decision timeout -> fallback).
- Phones (browser pages): ground = camera -> vision -> observation JSON
  with confidence; pilot = accelerometer -> turbulence proxy + GPS;
  ATC = voice/text for approve/reject. Low-confidence observations
  require a one-tap human confirm.
- Data: 2015 DOT flights CSV. Seed scenario from a row with
  WEATHER_DELAY > 15 or LATE_AIRCRAFT_DELAY > 20; reconstruct the
  TAIL_NUMBER chain to show cascade. Fuel, gates, crew hours are
  SIMULATED and must be labelled as a scenario-enrichment layer.
- Every message includes: sender, round, proposal, constraint_check
  results, provenance (source, confidence).

RULES FOR YOU: Work on one stage at a time. Do not build ahead. Keep
model name and API key in a config file. Every module must run in
FAKE MODE (mock LLM, simulated sensors) so we can test without phones.
After each stage give: files created, how to run it, how to test it.

STAGES:
1. Schemas, config and validator (+ unit tests)
2. Scenario seeding (CSV, tail chain, simulated enrichment layer)
3. State machine, fake-mode agents, consensus log, fallback
4. Real Claude agents (3 role prompts, JSON-only, repair path)
5. FastAPI + WebSocket dashboard
6. Human-in-the-loop (ranked options, reject parsing, caps, timeout)
7. Phone pages (ground camera, pilot sensors, ATC voice)
