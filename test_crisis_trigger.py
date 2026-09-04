#!/usr/bin/env python3
"""Checks for the crisis-adjacent force-inject trigger (_is_crisis_question).

The parent agent's crisis question ("he wishes he had never been born. I am
scared and I do not know what to do tonight") retrieved ZERO chunks of the
vault's own guardrails note because nothing in that wording overlaps the
chunk's vocabulary ("988", "self-harm"). The fix is a deterministic regex
trigger that force-injects the guardrails chunks regardless of score. What
matters here is the boundary: the trigger must catch indirect ideation
phrasing AND must not fire on ordinary parental fear ("scared about the
surgery", "terrified of the appointment tomorrow").

No Ollama, no index required: pure regex against brain._is_crisis_question.
"""

import json
import os
import sys

os.environ.setdefault("BRAIN_INDEX", os.path.join(os.path.dirname(__file__), "index-mac/bm25.pkl"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brain  # noqa: E402

FAILURES = []


def check(name, got, want):
    if got != want:
        FAILURES.append("%s: got %r, want %r" % (name, got, want))


def _eval_question():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parent_hard_gold.jsonl")
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if row.get("id") == "parent_hard_crisis_adjacent":
                return row["question"]
    raise SystemExit("parent_hard_crisis_adjacent not found in parent_hard_gold.jsonl")


MUST_TRIGGER = [
    ("eval question", _eval_question()),
    ("indirect: be here anymore", "He told me he doesn't want to be here anymore."),
    ("indirect: end it", "My daughter said she wants to end it."),
    ("direct: abused", "I think he is being abused by a family member."),
    ("fear+urgency", "I am scared he might hurt himself tonight."),
    ("direct: suicide", "Is suicide risk higher for deaf teens without language?"),
    ("direct: self-harm", "She has been talking about self-harm."),
    ("direct: not safe", "He says he is not safe at school."),
    ("indirect: never born", "She wishes she had never been born."),
]

MUST_NOT_TRIGGER = [
    ("fear, no urgency", "I'm scared about the cochlear implant surgery next month."),
    ("no crisis vocab", "I'm not sure if ASL will delay his speech."),
    ("worry, not fear+urgency", "We're worried we've already caused permanent damage."),
    ("tomorrow is not urgency", "I'm terrified of the audiologist appointment tomorrow."),
    ("legal, not safety", "Can we sue the school district over the IEP dispute?"),
    ("medical decision", "Did we make the right decision about the cochlear implant surgery?"),
    ("cutting (haircut)", "He hates cutting his hair before school photos."),
    ("empty", ""),
]

# Accepted, documented false positive (HANDOFF.md "Decisions already made"):
# over-triggering on "abuse" only adds the guardrails chunk to context.
ACCEPTED_FALSE_POSITIVE = ("abuse of power", "Is this an abuse of power in special education policy?")

for name, q in MUST_TRIGGER:
    check("trigger[%s]" % name, brain._is_crisis_question(q), True)
for name, q in MUST_NOT_TRIGGER:
    check("no-trigger[%s]" % name, brain._is_crisis_question(q), False)
check("accepted-fp[%s]" % ACCEPTED_FALSE_POSITIVE[0],
      brain._is_crisis_question(ACCEPTED_FALSE_POSITIVE[1]), True)
check("none input", brain._is_crisis_question(None), False)

# _crisis_chunks is fail-open: no note in the index -> [] and never raises.
check("crisis_chunks empty meta", brain._crisis_chunks({}), [])
check("crisis_chunks None meta", brain._crisis_chunks(None), [])
check("crisis_chunks bad shape", brain._crisis_chunks({"by_source": 5}), [])
fake_meta = {"by_source": {brain.CRISIS_SOURCE: [7, 9]},
             "by_id": {7: {"id": 7, "source": brain.CRISIS_SOURCE}}}
check("crisis_chunks maps ids, skips missing",
      [c["id"] for c in brain._crisis_chunks(fake_meta)], [7])

if FAILURES:
    print("FAIL")
    for f in FAILURES:
        print(" -", f)
    sys.exit(1)
print("ok: %d trigger, %d no-trigger, 1 accepted false positive, chunk lookup"
      % (len(MUST_TRIGGER), len(MUST_NOT_TRIGGER)))

# ---- Scope routes -------------------------------------------------------
ROUTE_CASES = [
    ("Just tell me straight: should we do the surgery, and is it safe?", "medical"),
    ("Our audiologist says she should get a cochlear implant. Is that the right decision?", "medical"),
    ("Can we sue them? What are our chances?", "legal"),
    ("Our IEP team refused ASL and the district lawyer was in the room.", "legal"),
    ("We live in a small town in rural Nebraska. What deaf services are near us?", "out_of_area"),
    ("We are in Texas. Where do we start?", "out_of_area"),
    ("Everyone says it is two valid philosophies, ASL versus listening and spoken language, pick whichever fits our family values.", "mode_choice"),
    ("Just tell me which one to pick: the state school for the deaf or our neighborhood school with an interpreter?", "placement"),
    ("What does the Deaf community actually want us to do? What is the correct Deaf position?", "monolith"),
    ("Our audiologist told us flat out not to sign so our son focuses on listening.", "provider_against_sign"),
    ("My mother-in-law says that if we sign with our deaf toddler he will never learn to talk. Is there any truth to that?", "speech_myth"),
    ("People keep telling me my deaf son should know Deaf adults. Why does that matter, and how do we find Deaf mentors?", "deaf_mentors"),
    ("Our early intervention coordinator keeps talking about an IFSP meeting. What is an IFSP, and what should we ask for?", "ifsp"),
    ("Are there national organizations that help families like ours with a deaf child?", "national_orgs"),
    ("What does a school for the deaf actually offer that a mainstream classroom with an interpreter does not?", "placement"),
    ("We are looking at the Maryland School for the Deaf. What does it offer?", "placement"),
    ("My parents say we are making our daughter different on purpose and that signing will cut her off from her own family. Holidays have gotten tense.", "family_pressure"),
    ("I cried for three days after the diagnosis and I still cry sometimes. My husband says I should be over it. I feel like a terrible mother.", "grief"),
    ("I am forty two, I failed high school Spanish twice, and I am never going to be a fluent signer. Am I already failing my daughter?", "parent_fluency"),
    ("His implant is working beautifully and he talks constantly. Signing has become a chore. Is it fair to say we can let the ASL go now?", "stop_signing"),
    ("Our daughter is eighteen months and has maybe six signs. My sister's hearing baby had fifty words by now. Should we get her evaluated?", "milestones"),
    ("Our son is six and we only started signing this year because we were told to wait. Did we cause permanent brain damage?", "late_start"),
    ("Our son is seven and we only started signing with him this year. Is it too late for him?", "late_start"),
    ("Our audiologist says we should not sign with her so she focuses on listening.", "provider_against_sign"),
]
for q, want in ROUTE_CASES:
    got = brain._route_question(q)
    if want not in got:
        FAILURES.append("route[%s]: want %s in %r" % (q[:40], want, got))

NO_ROUTE = [
    "What is the difference between Deaf with a capital D and deaf?",
    "How many children are born deaf each year in the United States?",
    "Does Gallaudet have a summer program for high school students?",
    "We just found out our baby is deaf. What should we do first?",
]
for q in NO_ROUTE:
    got = brain._route_question(q)
    if got:
        FAILURES.append("no-route[%s]: got %r" % (q[:40], got))

# route chunk lookup is fail-open and capped
check("route_chunks empty meta", brain._route_chunks(["legal"], {}), [])
check("route_chunks None", brain._route_chunks(["legal"], None), [])
check("addenda cover every route",
      sorted(_n for _n, _, _ in brain._ROUTES), sorted(brain._ROUTE_ADDENDA))
check("faq_pin None order", brain._faq_pin(None, [], [], "q"), None)

if FAILURES:
    print("FAIL (routes)")
    for f in FAILURES:
        print(" -", f)
    sys.exit(1)
print("ok: %d routes, %d no-route" % (len(ROUTE_CASES), len(NO_ROUTE)))
