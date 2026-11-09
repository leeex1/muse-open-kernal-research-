# OKC archive — how we competed (Nov 2026)

86 submissions across all 5 problems. This folder holds the **workflow representatives**, not
all 300 one-off variant scripts (those were parameter-sweep exhaust — see the learnings doc
for why that approach plateaued).

## `automation/`
| File | What it shows |
|------|---------------|
| `submit_driver.py` | The working submit loop: copy → `okc submit --no-wait --json` → parse job_id |
| `check_limits.py` | Daily scored-run usage per problem (20/day, resets 00:00 UTC) — check before burning runs |
| `poll_leaderboards.sh` | One-shot standings dump for all 5 boards |
| `modify_example.py` | The sed-replace pattern for parameter variants (verify hash!), with the PowerShell/WSL quoting fix (`/mnt/c/...`) |

## `baseline/`
The 5 vjeux canonical kernels we started from (`muse_*_vjeux.py`) — one per problem, verbatim.

## The workflow (condensed)
1. `okc leaderboard <problem>` → record leader speedup
2. Copy baseline → edit ONE parameter → hash-check the diff
3. `okc submit <problem> <file> --no-wait --json` → record job_id
4. `okc jobs` → wait for `passed` + time → compare
5. Repeat within the 20/day budget; batch near the 00:00 UTC reset

Final standings: #261 / #277 / #358 / #437 / #482. Full postmortem: `../guides/okc-contest-learnings.md`.
