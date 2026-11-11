# Local Pi workspace startup diagnostic

## Workspace-startup correction

The first Pi measurements above launched from `~/.pi/agent`. A later user report
started Pi from `~/Workspaces`, where the root `AGENTS.md` had accumulated 39,114
bytes of chronological device notes. Pi automatically included that file in its
system prompt. The captured request had **15,599 input tokens**, nearly the whole
16K context, and requested another 2,048 output tokens. Two user attempts were
cancelled after roughly 75 and 51 seconds without visible answer text. The server
sent keepalives; tool-enabled answers are buffered until generation finishes.
This did not establish a server deadlock.

The complete notes were preserved in a separate device-notes file, retaining all
historical records. The automatically loaded `AGENTS.md` now contains 1,802 bytes
of current operational instructions and a pointer to the full notes. The exact
question's prompt fell to **4,400 tokens** and completed in **23.94 seconds**;
the following France question took **0.75 seconds**. No runtime restart or
inference changes were required for this correction.

A separate fresh Pi launch from `~/Workspaces` used the unambiguous MLC/Nemotron
question. Three unrelated short completions first evicted ordinary conversation
entries, leaving only the protected server prefix reusable:

| Turn | Total (s) | New input | Cached input | Output |
| --- | ---: | ---: | ---: | ---: |
| Nemotron/MLC benchmark | 27.43 | 4,194 | 342 | 178 |
| France immediately afterward | 0.68 | 19 | 4,713 | 4 |
| Undo `git add .` | 1.32 | 17 | 4,735 | 18 |

The benchmark answer reproduced all three prefill/plain/speculative rows correctly.
The resident model was warm; this is a cold **conversation cache**, not a reboot
or disk-load measurement. These remain single latency samples. Record launch
directory and automatically loaded instructions when reproducing client timings.
Keep long memos outside auto-loaded instruction files and load relevant sections
when needed; keep required project instructions active.

The original question combined little-gemma V2 with Nemotron/Xavier. That pairing
is not established by the retrieved Orin results. A subsequent mixed-history
answer also mislabelled one benchmark row; those failures are recorded separately
from the two correct fresh Nemotron answers. This latency fix is not a general
answer-quality qualification. See
[`workspace-startup-summary.json`](../benchmarks/xavier/workspace-startup-summary.json).

