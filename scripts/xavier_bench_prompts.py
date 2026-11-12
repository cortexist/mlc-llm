"""Fixed prompts from the recorded Nemotron Xavier benchmarks."""

DESIGN_NOTE = (
    "Here is a short design note. Summarize it in three bullet points, then list two risks it "
    "does not mention.\n\nWe plan to move the nightly batch job that rebuilds the search index "
    "from a single large VM to a fleet of small spot instances. Each instance will take one shard "
    "of the document corpus, rebuild that shard, and upload it to object storage. A coordinator "
    "waits until all shards are uploaded, then atomically flips the index alias to the new "
    "version. If any shard fails three times, the coordinator aborts and keeps the old index "
    "live. We expect the job to finish in 40 minutes instead of 3 hours and to cost about a third "
    "as much. Monitoring will alert if the alias has not changed for 36 hours."
)

FILLER = (
    "The quick brown fox jumps over the lazy dog while the committee reviews the quarterly "
    "budget, the weather shifts from rain to sun, and the engineers rebuild the cache layer. "
)

# Added for the Qwen3.5 speculative decoding runs: answers of a hundred tokens and more.
CODE = (
    "Write a Python function that merges overlapping intervals in a list of (start, end) "
    "tuples. Include a short docstring and two example calls."
)
PROSE = "Explain in about 150 words why the sky is blue and why sunsets are red."
PLAN = (
    "I have meetings on Monday 9-11 and Wednesday 14-16 and need three two-hour focus blocks "
    "this week between 9 and 17. Propose a schedule and explain the choices briefly."
)
