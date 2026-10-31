# The human controller step

After the agents finish, a human controller decides. The agents propose; the validator checks; **a person
approves**. Nothing is "executed" without that approval, except by the clearly labelled fallbacks below.

## The flow

```
agents negotiate (max_rounds)                     the existing engine, unchanged
        |
        v
ranked options for the controller                 1 or 2, each re-checked by the validator
        |
        +-- approve option N      -> the validator re-checks it one last time -> APPROVED
        +-- reject, with free text -> turned into hard rules -> the agents negotiate again (a new "attempt")
        +-- no answer in time     -> our conservative default procedure (TIMEOUT_FALLBACK)
```

If the agents themselves end in the fallback, the controller is shown that fallback as option 1. If no safe
resolution exists at all, nothing is shown and the outcome is ESCALATED: a human must decide by other means.

## The options

- **Option 1** is what the agents agreed (or the conservative default procedure, if they did not agree).
- **Option 2** is the best valid alternative: the same total delay with the hold taken on the ground instead,
  otherwise the conservative default procedure.
- Each option shows its numbers side by side: total delay, fuel above the required minimum, crew-duty room,
  runway water margin and turbulence margin, plus a sentence saying how it differs from option 1.
- The options are produced by code from the agreed proposal, not written by a model. A model-written
  option would have to be validated anyway; this keeps what the controller sees fully checkable.

## Rejecting

The controller writes what to change, for example *"avoid gate A1 and hold at most 10 minutes"*. That text is
parsed into **structured hard rules** (a field, an operator and a value). They are added to the scenario and the
validator enforces them on every later proposal, every ACCEPT, and the fallback.

Limits (from `[limits]` in the config; all bounded):

| Setting | Default | Meaning |
|---|---|---|
| `max_human_rejections` | 2 | rejections that each trigger a renegotiation. A **third** rejection applies the fallback (still obeying every rule). |
| `decision_timeout_s` | 60 | seconds to answer each question before the fallback applies |
| `MAX_UNCLEAR` | 3 | answers that could not be understood, in a row, before the fallback applies |

Text that cannot be turned into a rule **does not use up a rejection**: the controller is told why, with
examples, and asked again.

## Why the free text is safe to act on

Two parsers share one gate (`negotiator/parsing.py`, `sanitize`):

- **Rule-based** (fake mode, tests): understands a small documented set of phrasings. Anything it does not
  understand is reported back, never guessed. Part of a sentence it cannot read is named, not silently dropped.
- **Claude** (live mode): structured output only. Its reply is treated as untrusted and goes through the same gate.

The gate allows only: the six plan fields, the six operators, values that **exist in this scenario** (a gate that
is not in the airfield data is refused), numbers in range, and at most 3 rules per rejection. A rule can only make
the validator stricter; it can never loosen a safety margin. The controller's text is data for the parser, not
instructions to it.

A rule can make the situation impossible (for example "avoid runway 27L" when it is the only dry runway). Then
there is no safe plan, the fallback finds none, and the outcome is ESCALATED. That is the correct, visible result.

## What is logged

One hash-chained log covers the whole session. New entry types: `OPTIONS` (what was shown), `DECISION`
(approve / reject / timeout, and the text), `CONSTRAINTS` (what the text was understood as, or why not),
`RENEGOTIATE`, and `FINAL` (the outcome and why). Each entry carries an `attempt` number, so round 1 of the
first negotiation and round 1 of the renegotiation are never confused. The dashboard shows them as
"Renegotiation 1, round 2". Logs written before this stage still verify.

## Final outcomes

| Outcome | Meaning |
|---|---|
| `APPROVED` | the controller approved an option, and the validator confirmed it again |
| `TIMEOUT_FALLBACK` | no answer in time |
| `UNCLEAR_FALLBACK` | the controller's answers could not be understood 3 times |
| `REJECTED_FALLBACK` | more rejections than renegotiations allowed |
| `VETOED_FALLBACK` | the approved option no longer passed the validator (for example a sensor changed while the controller decided) |
| `ESCALATED` | no safe resolution exists |

## Limits of this stage

- The controller is a browser (or the terminal). Voice input and phone pages are Stage 7.
- A low-confidence sensor reading still has no one-tap confirm; that arrives with the phone pages. Until then an
  unconfirmed reading simply blocks the options that depend on it.
- The rule parser in fake mode covers a fixed set of phrasings. The Claude parser is broader, but has had only
  mocked tests so far; run one live controller session before relying on it.
- The fake ATC bends its proposals to the controller's rules where it can; a real ATC agent is told about the
  rules in its prompt. In both cases the validator, not the agent, enforces them.
