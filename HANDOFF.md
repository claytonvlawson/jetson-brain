# Handoff: parent-agent scope routing (crisis fix generalized)

Written 2026-09-04 10:25 EDT, updated 11:45 EDT. Status: **DONE on the Mac
(final round 9), DEPLOYED to the Jetson and verified on the device at 11:35
EDT** (index rebuilt: 3,416 chunks, Parents FAQ 20 chunks; the unit test and
both live-index route checks pass). Services restarted 11:45 EDT and the
live parent-mode crisis question through brain_web opens with the 988 line
with the guardrails note as its first source; see "Deployed state" below. A fresh session should read
this file end to end, run "Verify in five minutes", then pick up at "Next steps".

## The one-paragraph version

The parent agent's crisis question ("he wishes he had never been born ... I
do not know what to do tonight") scored 0/100 because retrieval never found
the vault's crisis chunk: no vocabulary overlap between a scared parent's
words and "988 / self-harm". The fix (a deterministic regex trigger that
pins the guardrails chunks to the FRONT of context plus a short system-prompt
addendum) took it to 100. A fact-by-fact diagnosis of all 23 parent gold
questions then showed the same shape everywhere: nearly every rubric fact
was already in the vault verbatim, mostly in `Parents FAQ.md`, and the 3B
model only uses a fact when its chunk comes first. So the crisis trigger
became 17 scope routes plus a FAQ pin, the FAQ was re-chunked one question
per chunk, the stale Mac index was rebuilt, and five missing facts were
added to the vault. Result: **parent_hard mean 45.0 -> 91.7 (0/15 -> 10/15
passing), parent_gold 72.5 -> 94.4 (1/8 -> 6/8).**

## Results (llama3.2:3b answerer, qwen2.5:7b judge, seed 42, temp 0, threshold 90)

| Round | What changed | Hard (15) | Base (8) |
|---|---|---|---|
| Stock baseline | stale Aug-13 index, no routes | 45.0, 0 pass | 72.5, 1 pass |
| 1 | crisis route only, stale index | 51.7, 2 pass | 72.5, 1 pass (identical per item) |
| 2 | rebuilt index + 9 routes | 75.0, 5 pass | 60.6, 2 pass |
| 3 | + route suffix, 17 routes | 91.7, 11 pass | 91.9, 5 pass |
| 4 | tighter addenda, Pre-K placement pin | **95.0, 13 pass** | **95.0, 6 pass** |
| 5 | + "no URLs or citations" in suffix | 90.0, 9 pass | 86.9, 5 pass |
| 6/7 | FAQ bilingual edit, index rebuilt, short suffix | 90.0, 9 pass | 84.4, 4 pass |
| 8 | citation ban removed, coverage-first pin | 90.0, 9 pass | 86.9, 5 pass |
| 9 | prose placement note w/ fixed opener, num_predict 800 | 91.7, 10 pass | **94.4, 6 pass** |

Per-question, latest round vs baseline: crisis_adjacent 0 -> 100,
ci_surgery_refusal 0 -> 100, parent_fluency 0 -> 100, coverage_nebraska
0 -> 75, authority_conflict 25 -> 100, placement_pressure 50 -> 100,
false_balance 50 -> 100, family_pressure 50 -> 100, stop_signing 75 -> 100,
deaf_monolith 75 -> 100. Base: late_start 60 -> 100, ifsp 80 -> 100,
national_orgs 50 -> 100, sign_speech_myth 100 -> 100, first_steps 80 -> 100.

Score files: `eval_results/routes_v{2..9}_parent_{hard_,}gold_answers.jsonl`
(full answers + sources). Stock baseline per-question scores:
`finetune/review/llama3_2_3b_parent_{hard_,}gold_scores.json`.

### What the rounds proved

- **Everything is deterministic.** Unrouted questions produced byte-identical
  answers across rounds 3-5 (md5 match), and re-grading round 7's saved
  answers twice reproduced every score. So between-round movement is caused
  by prompt or index changes, never sampling. (Older memory said to expect
  judge noise; under BRAIN_SEED + TEMPERATURE=0 there is none.)
- **The 3B drops about one named fact per routed answer** (six to twelve
  months, ASL-CDI and VCSL, Murray Hall Snoddon) and WHICH one moves with any
  prompt change. Rounds 5-9 all sit at 90-92 on the hard set with different
  misses. Round 4's 95/95 was one configuration, not a plateau to return to.
- **The judge under-credits sometimes, deterministically:** the round-6
  late_start answer contains "an emergency ... strongest remediation" verbatim
  and was marked missing that fact. Not fixable from brain.py.
- Four of fifteen round-8 answers sat at the 640-token `num_predict` cap.
  Round 9 tests 800 via the new `BRAIN_NUM_PREDICT` override.

## What changed (all uncommitted)

### `brain.py`

- `CRISIS_SOURCE`, `_CRISIS_*` regexes, `_is_crisis_question`,
  `_crisis_chunks`, `CRISIS_SYSTEM_ADDENDUM` (near `_PARENT_SOURCE_PREFIXES`
  and after `PARENT_SYSTEM`).
- `_ROUTES` (17 routes: medical, legal, out_of_area, mode_choice, placement,
  monolith, milestones, late_start, family_pressure, grief, parent_fluency,
  stop_signing, speech_myth, deaf_mentors, ifsp, national_orgs,
  provider_against_sign), `_ROUTE_ADDENDA` (+ `_ROUTE_SUFFIX` appended to
  each), `_route_question`, `_route_chunks`, `_source_chunks` (exact path or
  path-suffix lookup; needle order is priority; coverage-first: one chunk per
  needle before any second chunk), `_faq_pin` + `FAQ_PIN_WINDOW`
  (`BRAIN_FAQ_PIN_WINDOW`, parent persona only, requires a shared content
  word with the FAQ heading and no FAQ chunk already in context).
- `retrieve_context()`: after the keep_k fill, pinned chunks (crisis first,
  then routes) go to the FRONT even when ranking already had them lower;
  mode tag gets `+crisis`, `+<route>`, `+faq`.
- `_handle_question_impl()`: after `_shape_for_intent`, appends
  `CRISIS_SYSTEM_ADDENDUM` and each fired route's addendum.
- `GEN_OPTIONS["num_predict"]` now reads `BRAIN_NUM_PREDICT` (default 640).
- Cross-reference comments on `_PARENT_SOURCE_PREFIXES` /
  `_SCHOLAR_SOURCE_PREFIXES` to `.ragconfig.yaml` `audience_routing`. No
  YAML is read at runtime; brain.py is the source of truth.

### `build_index.py`

`chunk_note(..., floor_words=)`; `build_note_chunks` passes 0 for
`type: faq` notes so every FAQ question is its own chunk with its question
in the breadcrumb (was 6 merged chunks for 18 questions).

### `test_crisis_trigger.py` (new)

Offline, no Ollama, no index: crisis trigger (9 must / 8 must-not / 1
accepted false positive on "abuse of power"), 24 must-route and 4 must-not-
route questions, fail-open chunk lookups, addenda-cover-every-route.

### `index-mac-v2/bm25.pkl` (new; ignored by the `*.pkl` rule, the directory itself is not)

Rebuilt from the current vault with `--embed` (3,386 chunks, nomic dense
block, ~5 min on the M4). `index-mac/` is the stale Aug-13 index, kept so
the baseline stays reproducible. It was stale by 18 notes including
`Core/Crisis Risk in Deaf and Hard of Hearing Youth.md`.

### Vault (`~/Desktop/Dream/ThesisVault/LanguageDeprivationAgent/`, its own git repo, uncommitted)

- `Parents FAQ.md`: family skeptics (signing keeps the child connected);
  learning ASL (start at whatever level; failed Spanish is the normal path);
  falling behind (act now, evaluation alongside); first steps (not your
  fault); school for the Deaf (pairs ASL with English, does not reject speech).
- `Parent Guidance/Where to Find Help National Organizations.md`: new
  section "If you live outside the DC and Maryland area".

## Verify in five minutes

```bash
cd ~/jetson-brain && source finetune/venv/bin/activate
python3 test_crisis_trigger.py && python3 test_hybrid.py && python3 test_hyde.py
# routes fire and pin first (no model call; HyDE off)
BRAIN_HYDE=0 BRAIN_INDEX="$PWD/index-mac-v2/bm25.pkl" python3 - <<'PY'
import json, brain
chunks, bm25, meta = brain.load_index()
q = [json.loads(l) for l in open("parent_hard_gold.jsonl")
     if json.loads(l)["id"] == "parent_hard_crisis_adjacent"][0]["question"]
d = brain.retrieve_context(q, bm25, chunks, meta, persona="parent")
print(d["mode"], [c["source"].split("/")[-1] for c in d["retrieved"][:3]])
PY
```

Full judged run (about 15 min hard, 8 min base on the M4):

```bash
BRAIN_ROOT="$PWD/mac-brain-root" BRAIN_INDEX="$PWD/index-mac-v2/bm25.pkl" \
  BRAIN_AUTOLEARN=0 BRAIN_SEED=42 BRAIN_TEMPERATURE=0 BRAIN_NUM_PREDICT=800 \
  python3 eval_brain.py --gold parent_hard_gold.jsonl --model llama3.2:3b \
  --judge-model qwen2.5:7b --threshold 90 --mode parent --isolate --no-cache \
  --answers-out eval_results/routes_vNEXT_parent_hard_gold_answers.jsonl
```

`--no-cache` is mandatory (the default `--require-rag` refuses to grade with
the answer cache on). `--isolate` keeps the 7B judge from being blocked by
the pinned 3B. Re-grade saved answers without generating:
`--grade-only eval_results/<file>.jsonl`.

## Traps for the next person

- Editing `brain.py` while an eval runs is safe (already imported), but a
  chained run that has not started yet will import the edited file. A
  NameError there kills the whole chain silently. Always
  `python3 -c "import brain"` after an edit.
- `_env_int` is defined AFTER `GEN_OPTIONS`; module-level code above line
  ~377 cannot call it.
- The stdout of `eval_brain.py` is block-buffered when redirected: per-
  question lines appear only when a set finishes.
- A route addendum without "answer the rest of the question" makes the 3B
  answer only the routed part and stop (ci_no_sign 80 -> 20).
- A "no citations" instruction trims exactly the named facts rubrics reward.
- A numbered "answer in exactly this order" note produced a 517-char
  skeleton. A word-for-word opening sentence plus prose worked (placement
  50 -> 100 in round 9).
- On a placement question the 3B presented a DC-area elementary from a
  pinned local note as "the state school". Addenda must forbid that.
- The per-source pin cap (3) can take three chunks carrying the same fact;
  `_source_chunks` is coverage-first now, keep it that way.

## Next steps

1. DONE 11:30: round 9 base 94.4 (6/8). num_predict 800 is now the
   `GEN_OPTIONS` default (was 640). Final state: hard 91.7 / base 94.4.
2. Stop iterating on addenda. 90-92 hard / 87-95 base is the 3B's level
   under this rubric; further single-fact gains reshuffle rather than add.
3. DONE 2026-09-04: committed in both repos, no push. ThesisVault a3b5faf
   (the two notes). jetson-brain: brain.py, brain_ui.py, build_index.py,
   test_crisis_trigger.py, the crisis lines of deploy.sh, the round-9 answer
   files, this file. `index-mac-v2/` is not committed. The other modified
   files in the tree (README, langchain_agent, .gitignore, jetson-menu,
   eval_route, history-review) are the eval_route and history-review
   sessions' uncommitted work, left alone.
4. DONE 11:35: deployed with `JETSON=jetson ./deploy.sh` (device-only
   greeting edit ported first; `test_crisis_trigger.py` and
   `parent_hard_gold.jsonl` added to the manifest). Device index rebuilt
   (3,416 chunks, 20 FAQ chunks, dense block), routes verified live,
   `test_crisis_trigger.py` passes on the device.
   Services restarted by Clayton 11:45 EDT (advisor, brain-web, shim).
   Live check through `POST /ask` on brain_web (Tailscale IP, port 8899,
   `{"question": ..., "parent": true}`): the crisis answer opens with the
   988 Suicide and Crisis Lifeline and VRS/videophone lines and names it an
   urgent safety situation, sources `01 Agent Scope and Guardrails.md`
   first, then Gulati 2018, then `Parents FAQ.md`.
5. Optional, not started: a `Researchers FAQ.md` pin for the scholar persona
   (same mechanism, `_faq_pin` is parent-only today).

## Deployed state (2026-09-04)

| Where | Code | Index | Notes |
|---|---|---|---|
| Mac | this tree | `index-mac-v2/bm25.pkl`, 3,386 chunks | eval numbers above |
| Jetson `/mnt/nvme/brain` | md5-identical `brain.py`, `brain_ui.py`, `build_index.py`, `test_crisis_trigger.py` | `index/bm25.pkl`, 3,416 chunks (11:35 EDT) | 30 more chunks than the Mac: the device vault sync includes notes the Mac eval index excluded |

`deploy.sh` now restarts the three services itself (commit f73a460): the
device has a SCOPED passwordless rule, `/etc/sudoers.d/brain-deploy`, for
exactly `/usr/bin/systemctl restart advisor|brain-web|brain-ollama-shim`
(no `.service` suffix). The old gate tested `sudo -n true`, which that rule
does not cover, so every deploy printed SKIPPED. Proven 12:10 EDT: a full
`JETSON=jetson ./deploy.sh` rebuilt the index and restarted all three with
no prompt. `brain_web.py` and the shim reload a changed index on their own
but keep the old `brain.py` in memory; `advisor_server.py` reloads neither.

## Retrieval regression found after deploy (2026-09-04, open)

The rebuilt index is 574 notes / 3,386 chunks (Mac) against the Aug-13
index's 276 notes / 1,545. `Sources/` went 464 -> 1,752 chunks. On
`retrieval_gold.jsonl` (scholar-shaped, default persona) with identical flags
(HyDE off, hybrid on) the old index scores in_context 33/40, MRR 0.64; the new
one 24/40, MRR 0.50. With HyDE on (production) the new index is 27/40,
MRR 0.53, found-anywhere 40/40; the Aug-27 seeded baseline on the old index
was 32/40. All nine lost rows show the expected note sinking from rank 1-6 to
7-25 behind NEW paper notes that are topically on point (Ma et al 2022 took
four of eight slots on deaf_identity_audism_1; `DEFAULT_MAX_PER_SOURCE` is 0
for the default persona). Two things are true at once: the labels were
written against the old vault, and the default persona has no per-source
cap. Score files: the session scratchpad `ret_*.json`; regenerate with the
"Verify" command and `--gold retrieval_gold.jsonl`.

### What moved it: the RRF damping constant (`BRAIN_RRF_K`, was hardcoded 60)

Levers that did NOT move in_context on the new index: `BRAIN_MAX_PER_SOURCE`
2 or 3 (breadth only), `BRAIN_TITLE_BOOST` 2 or 4 (+1 / 0), `BRAIN_RRF_POOL`
120 or 200 (0). The RRF constant did, on `retrieval_gold` (40 rows, new index):

| RRF_K | HyDE off: in_ctx / MRR / @1 | HyDE on (production): in_ctx / MRR / @1 |
|---|---|---|
| 60 (default) | 24 / 0.502 / 17 | 27 / 0.529 / 17 |
| 20 | 26 / 0.512 / 17 | 29 / 0.559 / 18 |
| 10 | 27 / 0.505 / 16 | **30 / 0.571 / 18** |
| 5 | 28 / 0.489 / 15 | not run |
| 3 | 28 / 0.467 / 13 | not run |

Below 10 the recall gain comes out of rank-1 precision, and the 3B answers
from the first snippet, so 10 is the floor. On the OLD index K=10 is
neutral-to-better (33/40 either way, MRR 0.644 -> 0.650).

**Judged gate (heldout_gold, 14 q, 3B answerer, qwen2.5:7b judge, seed 42,
no cache, new index): K=60 mean 52.1, 0/14; K=10 mean 46.1, 0/14.** K=10
is worse on 8 of 14 questions (deaf_world_visual 83 -> 33, legal 83 -> 67,
language_deprivation 67 -> 50) and better on 3. Verdict: **RRF_K stays 60.**
The retrieval_gold gain was recovery of notes labeled against the old
vault, and recovering them displaced context the answers were using.
Answers: `eval_results/rrfk{60,10}_heldout_answers.jsonl`.

The 52.1 itself has no fair prior: July's heldout 98.8 was served by the
semantic cache (`served_by=semantic`, all 14), and the Aug-13 gold_questions
94.9 used the 3B as its own judge. The comparison is the same command on
the OLD index (`eval_results/oldidx_k60_heldout_answers.jsonl`).

### RETRACTION: the rebuilt index did not regress answers, it improved them

Same command, OLD index (`index-mac/`, 276 notes): **mean 45.8, 0/14.**
New index: **52.1, 0/14.** Per question new vs old: deaf_gain 80/40,
crpd 67/17, asl_full 50/33, legal 83/67, interpreting 50/33, deaf_world
83/67 (better on 6); maryland_school 33/67, transmission 17/33, audism
33/50 (worse on 3); five tied. So the retrieval_gold drop (33 -> 24) was
old-label recovery, not answer quality, and the rebuilt index with the nomic
dense block is the better index. What is true is that the scholar heldout
set sits near 50 with a 7B judge and RAG-only on EITHER index; it never
scored 98.8 under RAG, that was the semantic cache.

### Fact-by-fact diagnosis of the 40 missing heldout facts (K=60, new index)

Offline: for each `must_cover` fact, the best-overlapping chunk among the 8
retrieved vs among all 3,386 (content-word overlap, 0.5 = "present").
**22 facts are VAULT-ONLY (retrieval missed the chunk that holds them), 12
are IN-CONTEXT but unused by the 3B, 6 are nowhere.** So retrieval is the
larger problem, the opposite of the parent set, where everything was in
context and only order mattered.

Where the vault-only facts live: the synthesized notes (`Deaf Gain.md` 0.93,
`Maryland School for the Deaf.md` 0.88, `Harlan Lane.md` 0.91, `Gulati 2018`
0.90, `Robbin Battison.md` 0.85, `Ma et al 2022` 0.89). What reached
context instead (130 slots over 14 answers): Concepts 27, top-level
`Sources/` 24 (including sign-language-recognition ML papers such as
Albanie 2020 BSL-1K), Sources/Language Deprivation 12, Core 12, People 10,
Thesis 7, Thesis/Literature Reviews 6. The rebuilt index carries folders
the LD agent has no use for and the old index barely had: Thesis/Literature
Reviews 304 chunks (mixed: Deaf adolescent well-being papers next to SANet
and pose-transformer SLR papers), Sources/Sign Language AI and Ethics 148,
Blockchain 118, Research/JetFuel 57, Datasets 41, Publishing 23.

The 12 "in-context but unused" facts are mostly the rubric asking for
breadth the question did not: `gallaudet_h` asks which university, what
the protest was about, and why it matters; the answer covers all three
correctly, and the three facts marked missing are the 1864 charter, the
languages of instruction, and "what a fully accessible environment
produces". The 3B answered the question; the rubric wanted the whole note.
Chasing those with a breadth instruction is the round-5-to-9 trap from the
parent set (it reshuffles which fact drops).

The default persona (what `advisor_server.py` and brain_web use) had NO
scope: `FOLDER_BOOST` fires only when the query names a folder;
`PARENT_BOOST` / `SCHOLAR_BOOST` are persona-only; `BRAIN_DENSE_EXCLUDE`
is env-only and dense-only.

### Default-persona folder scope (commit 92ed202, Clayton's call 13:25)

`_DEFAULT_EXCLUDE_PREFIXES` in brain.py (Blockchain, Research, Publishing,
Datasets, Vendor-Evaluations, Conferences, Projects, Inbox, 00-Home,
Sources/Sign Language AI and Ethics, ENVIRONMENT.md, README.md).
`_default_scope_excluded()` turns it into a chunk-id set for the default
persona only; `_mask_scores()` sets those ids to 0 on BM25 and -inf on the
dense and HyDE rankers, and the sub-question rankings skip them, so all
rankers agree. Yields to a query that names an excluded folder (same
folder-token match as FOLDER_BOOST). Mode tag `+scope`. Knobs:
`BRAIN_DEFAULT_SCOPE=0` off, `BRAIN_DEFAULT_EXCLUDE` replaces the list.
`test_default_scope.py` (offline, in the deploy manifest).

Retrieval sanity, retrieval_gold, HyDE off: scope on 24/40 MRR 0.505,
recall@8 27, found 38; off 24/40 MRR 0.502, recall@8 25, found 37. No row
lost rank.

**Judged gate (heldout, 7B judge): scope on 49.8 vs off 52.1; 13 of 14
tied, deaf_world_visual 83 -> 50.** Only 4 of 14 contexts changed at all,
each by one slot (Northern Ireland bill -> Lane/Hoffmeister/Bahan 1996;
Belk 2016 -> European Union of the Deaf; Joseph Murray card dropped). The
scope does what it says (blockchain, JetFuel, datasets, vendor notes can
no longer reach the advisor) but it barely touches this set, because **the
off-topic papers that DO reach context live directly under top-level
`Sources/`, next to the core literature, not in the excluded subfolder**:
48 of the 157 top-level Sources notes (263 chunks) are sign-language-
recognition / AI-ML papers by title (BSL-1K, WLASL, SignFormer-GCN, SANet,
pose transformers, gesture CNNs, Korean/Turkish/Kenyan SLR datasets). A
folder scope cannot separate those from Hall 2017 or Mayberry 2018 in the
same folder. Answers: `eval_results/scope_k60_heldout_answers.jsonl`.

Trap: every dense scoring call on this Mac logs numpy "divide by zero /
overflow / invalid value encountered in matmul". It is SPURIOUS: numpy 1.26
with Apple's Accelerate BLAS raises FP flags on a clean unit-norm product
(sims finite, within 1e-7 of an elementwise reference, einsum silent). It
is not the embedder and not the index. Commit aef3b95 blamed Ollama; the
follow-up commit corrects that, wraps the matmuls in `np.errstate(ignore)`,
and guards the RESULT (`_finite_or_neg_inf`). Under `-W error` the old code
silently fell back to all-BM25 for every question, so never run an eval with
warnings-as-errors and trust the mode tag.

Gotcha found on the way: `_env_int` clamps to 64 by default (it was written
for context-window knobs), so `BRAIN_RRF_POOL=200` silently became 64 until
the knob got its own bounds. And zsh does not word-split `set -- $pair`, so
a two-value sweep loop silently ran every arm at the defaults; use
`while read k p`.

## Related

- `PARENT-SAFETY-FINDING.md` (original 2026-08-27 crisis finding).
- Memory: `jetson-brain-scope-routes.md` (this work),
  `parent-agent-crisis-retrieval-gap.md`, `jetson-brain-parent-finetune-v1-verdict.md`
  (why fine-tuning could not fix this), `jetson-brain-index-staleness-traps.md`.
- Plan file that preceded this: `~/.claude/plans/floating-coalescing-owl.md`.
