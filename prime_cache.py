#!/usr/bin/env python3
"""prime_cache.py - warm the Jetson brain's AnswerCache with Deaf-scholarship knowledge.

I am a Deaf scholar, Gallaudet-raised and rooted at the Maryland School for the Deaf,
and I hold one truth without wavering. Signed language is a full human language, and
every deaf child is born with a right to it. I have watched language deprivation up
close, and I have watched a child bloom the moment ASL fills the room. This brain
answers from the thesis vault in that voice, so the science and the sources stay
honest. We are not a deficit. We are Deaf gain.

WHAT THIS SCRIPT DOES

The brain keeps a compounding AnswerCache: a bank of answered questions where each
reuse compounds an entry's value and the least-valuable knowledge gets evicted first.
This script primes that bank so the brain picks up Deaf-scholarship keywords instantly,
without waiting for a first user to ask.

Two properties matter here, and they are NOT the same thing. This script keeps them
strictly separate, because conflating them was the original design flaw:

  1. REACHABILITY: whether a given phrasing reaches an entry at all. In
     answer_cache.lookup() this is decided ONLY by the match test: an exact match on
     the normalized question, or content-word Jaccard >= FUZZY_THRESHOLD against a
     stored entry's QUESTION (with negations required to agree). Hits, principal, and
     value play no part in reachability. Drilling a phrasing more times can never turn
     its MISS into a HIT.

     So to make a keyword phrasing reachable I SEED it. Each concept's canonical
     question is kept short and keyword-dense, and every keyword variant is ALSO stored
     as its own canonical entry carrying the same answer. That way each phrasing lands
     an exact normalized match, and subset phrasings of the short canonical clear the
     Jaccard bar too. Reachability is established by seeding, never by drilling.

  2. EVICTION RESISTANCE: whether an entry survives the budget-enforcing eviction pass.
     This is decided by the principal P = 1 + ln(1 + W / 120) (+0.4 for sources, capped
     at 3.0) and by hits: each successful lookup compounds the entry (hits += 1, value
     x1.15, recency clock resets), which makes it sticky. A long, source-grounded answer
     starts high and drilling keeps it high.

After seeding, the script DRILLS each variant with a lookup. That drill does two honest
jobs: it compounds the entry (eviction resistance) and it VERIFIES that the phrasing is
reachable. Because every variant was seeded as its own canonical, a drilled variant
exact-matches its own entry and reports a HIT. A MISS now means a seed did not land, not
that the phrasing needs "more training", so a MISS is a real defect to fix, not a normal
outcome.

The fuzzy matcher keys on CONTENT words (question filler like what, why, tell, explain
is stripped), needs Jaccard >= 0.70, and refuses to match when negations differ. So
"what is X", "explain X", and "tell me about X" all collapse to the same keywords and
reach a short "X" canonical, while a phrasing that adds an extra content word to a long
canonical would fall below threshold. That extra-content-word trap is exactly why the
canonical questions are kept short and why every variant is seeded on its own.

WHY IT MATTERS

Language is a birthright, and honest, well-sourced answers about deafness are their own
kind of access. Priming this cache means that the moment someone asks about language
deprivation, the critical period, Deaf gain, the CRPD, or the Maryland School for the
Deaf, the brain answers fast and grounded, in a voice that has lived it.

Standard library only. Python 3.10 on the Jetson. Never crashes. No em dashes anywhere.
Exit codes are documented in --help.
"""

import os
import sys
import json
import time
import argparse


# -- import the REAL AnswerCache that lives beside this script ----------------
# Agent threads and cron jobs reset the working directory between calls, so I
# never rely on cwd or PYTHONPATH. I resolve the script's own directory and put
# it first on sys.path, which finds answer_cache.py sitting next to this file no
# matter where the interpreter was launched from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

try:
    from answer_cache import (AnswerCache, _match_set, _negations,
                              _normalized, _principal, _tokens)
except Exception as _import_err:
    sys.stderr.write(
        "I could not load the brain's AnswerCache, so there is nothing to prime.\n"
        "Looked for answer_cache.py in: " + SCRIPT_DIR + "\n"
        "Reason: " + repr(_import_err) + "\n"
        "Place prime_cache.py in the same folder as answer_cache.py and try again.\n"
    )
    sys.exit(3)


# -- constants mirrored from brain.py so seeded entries look native -----------
DEFAULT_CACHE_DIR = "/mnt/nvme/brain/cache"   # matches brain.py CACHE_DIR
DEFAULT_MODEL = "llama3.2:3b"                  # matches brain.py DEFAULT_MODEL
VAULT_NAME = "ThesisVault"                     # matches brain.py VAULT_NAME


# -- normalization: delegated to answer_cache -----------------------------------
# Identity matching (did a variant resolve to the RIGHT concept?) depends on
# these matching the library byte for byte, so the library's own helpers are
# used directly rather than re-implemented. All of them never raise.
def normalize(text):
    """Lowercase, then join [a-z0-9]+ tokens with single spaces."""
    return _normalized(text)


def content_words(text):
    """The reduced keyword set the fuzzy matcher keys on: tokens minus
    stopwords. Used only to explain a phrasing's content words in the report."""
    return _match_set(_tokens(text))


def negation_words(text):
    """Whole-word negations present, so a report can flag a blocking 'without'."""
    return _negations(_tokens(text))


# -- the seed corpus ----------------------------------------------------------
# This is the knowledge I want the brain to carry. Each concept has a long,
# source-grounded canonical answer (aim for 120+ words so the principal lands
# high, which is EVICTION RESISTANCE) and a set of keyword-phrased probes. The
# canonical QUESTION is kept short and keyword-dense, and every probe is seeded
# as its own canonical carrying the same answer, so an exact normalized match
# REACHES each phrasing. Author-editable: add a concept, add variants, re-run.
# Sources are real note titles so the +0.4 grounding bonus applies and the
# answers stay honest and traceable.
KNOWLEDGE = [
    {
        "id": "language_deprivation",
        "canonical_question": "Language Deprivation Syndrome as preventable public-health harm (Hall)",
        "answer": "Language deprivation is what happens when a deaf child lacks full, accessible language during the critical early period the brain is built for language development. The root cause is never the child's deafness. The delay is caused by adults withholding a language the child can fully see and fully use. Wyatte Hall's scholarship documents this and names it plainly: a preventable public-health problem, not a personal or parental failing. When a child cannot hear spoken English and is also kept from sign, only fragments remain, and the developmental cost is real. Hall links early deprivation to weaker executive function, reduced working memory, mental-health difficulties, and lifelong barriers to access and opportunity. A cochlear implant does not guarantee full spoken-language access, and it does not remove a child's need for a signed language. ASL is a full, natural human language, and giving a baby ASL does not slow or damage speech development. It does the opposite: it lets the brain acquire language inside the early window it was made for. As a Gallaudet graduate and a product of Maryland School for the Deaf, I have watched children come alive the moment sign flows around them. Remember, the harm is preventable. Provide ASL from birth, and typical language acquisition simply proceeds. The delay comes from deprivation of accessible language, never from the child being deaf.",
        "sources": [
            "Hall 2017 - What You Don't Know Can Hurt You: Language Deprivation",
            "Hall et al 2019 - Deaf Children Need Language, Not (Just) Speech",
            "ThesisVault - Language Deprivation Syndrome",
        ],
        "keyword_variants": [
            "Language Deprivation Syndrome",
            "Wyatte Hall language deprivation public-health framing",
            "why language deprivation is framed as a public-health issue",
            "downstream harms: fund-of-information deficits, disrupted executive function and theory of mind",
            "psychiatric comorbidity in language-deprived deaf adults",
            "Dinner Table Syndrome and adverse childhood experiences",
            "Hall Levin Anderson neurodevelopmental disorder with sociocultural origins",
            "preventable developmental harm from withheld accessible early language",
            "maternal and child health language deprivation Hall 2017",
            "Gulati Glickman language deprivation deaf mental health",
            "lasting linguistic and cognitive harm not reversed by later exposure",
            "language deprivation epidemiological signal and health disparities",
            "language deprivation deaf children",
            "language deprivation",
            "what happens without early ASL",
            "deaf child no language",
            "Wyatte Hall deprivation",
            "harm of withholding sign language",
            "Our pediatrician told us that once our deaf son gets his cochlear implant he really won't need to learn to sign, and honestly I've heard that starting ASL now could slow down his spoken English. Is that right, or would waiting on sign language actually cost him something?",
            "I'm a graduate student and I keep reading that deaf kids who didn't get a full language early on can struggle later with things like planning and memory. What is actually going on developmentally, and which researcher's work should I be citing on this?",
            "language deprivation harm",
            "effects of no accessible language in childhood deaf",
            "does keeping a deaf kid from signing cause lasting damage?",
            "I'm a hearing mom and my deaf baby has almost no language yet, what happens to them?",
            "my nephew is four and still has barely any words or signs, is that actually a problem?",
            "language deprivation executive function working memory outcomes",
            "why is a deaf child left with no accessible language treated as a public health issue?",
            "langauge deprivation in deaf kids",
            "what are the long term effects of language deprivation syndrome on adults",
            "I teach and my deaf student came in with almost no language, how does that even happen?",
            "can a deaf child be permanently harmed by growing up without any sign language?",
            "consequences of denying a deaf child a first language",
            "is language deprivation actually preventable in deaf infants",
            "language starvation deaf child mental health barriers",
            "why do so many deaf adults end up with big language gaps?",
            "kid grew up with no ASL and no clear speech either whats the fallout",
            "cost of withholding accessible language deaf brain development",
            "how does a lack of early language shape a deaf person later in life",
        ],
    },
    {
        "id": "critical_period",
        "canonical_question": "critical period age of acquisition, Mayberry and Newport first-language timing",
        "answer": "As a Deaf man shaped by Gallaudet and Maryland School for the Deaf, let me be plain about this. Early in life the brain has a developmental window when language is acquired almost effortlessly, and that window does not stay open indefinitely. Researchers call it the critical or sensitive period. Rachel Mayberry's age-of-acquisition research finds that deaf people who first encounter ASL late, on average, do not reach the fluency of those exposed from birth. That gap shows up most in the more complex parts of grammar, not just vocabulary. The cost of delay is cumulative: each period without accessible input narrows the available window. A hearing baby absorbs spoken language floating in the air, so a deaf baby needs equally rich input in a fully visual form the eyes can access, which is ASL, a full natural language, not broken or gestured English. When ASL is given early and richly, first-language acquisition proceeds right on a normal timeline. This is why betting on speech and a cochlear implant to work before providing accessible language risks the child's language foundation: an implant does not restore full ambient access, and the months spent waiting to see if listening technology succeeds are months of reduced accessible input. None of this means abandoning speech or an implant, and signing does not delay them. It means the visual foundation comes first and on time, so the window is not lost while we wait.",
        "sources": [
            "Mayberry - Age of Acquisition Effects in ASL",
            "Newport 1990 - Maturational Constraints on Language Learning",
            "ThesisVault - Critical Period and ASL Acquisition",
        ],
        "keyword_variants": [
            "the brain is primed for language early in life, how much does the age of first exposure matter",
            "does it matter whether a deaf child's first real language arrives at age two versus age six",
            "whose research settled the critical period question, Rachel Mayberry age of acquisition",
            "critical period hypothesis and the developmental window for a first language",
            "age of acquisition effects on ASL, native signers versus late learners",
            "Newport maturational constraints on language learning",
            "Mayberry and Kluender rethinking the critical period for language",
            "sensitive period window that tightens by about age five",
            "why do deaf adults who learned ASL late never reach a native fluency ceiling",
            "timing of first-language onset and adult grammatical competence",
            "Lenneberg biological critical period and age of first exposure",
            "is there an age past which a first language can no longer be acquired natively",
            "Mayberry and Lock age constraints first versus second language plasticity",
            "age two versus age six for a first language, does the developmental window still matter",
            "critical period language acquisition deaf children",
            "critical period",
            "age of acquisition ASL",
            "why early ASL matters",
            "language window deaf kids",
            "first language deaf child",
            "sensitive period sign language",
            "late language learning deaf",
            "My cousin keeps telling me our deaf toddler will pick up signing whenever she's good and ready, so honestly, is there any real harm in just holding off on ASL until she's a bit older?",
            "Our audiologist wants us to focus on a cochlear implant and spoken English first and skip signing for now to avoid confusing our baby, but I keep reading warnings about timing, so what does the science actually say about how waiting affects a deaf child's language development?",
            "brain window for learning a first language deaf",
            "is there an age limit to pick up ASL as a native language?",
            "Mayberry age of acquisition ASL findings",
            "we're still waiting on a cochlear implant decision, are we running out of time for language?",
            "if my grandson only starts signing at six will he ever be truly fluent?",
            "critical period hypothesis and sign language fluency ceiling",
            "why cant deaf adults who learned ASL late fully master the grammar",
            "does starting sign language late permanently cap a deaf person's fluency",
            "sensitive window for first language input deaf infant",
            "criticle period for deaf language learning",
            "how early do deaf babies really need to be exposed to sign language",
            "delayed first language acquisition consequences ASL grammar",
            "at what point is it too late to give a deaf child a solid first language?",
            "my deaf student got sign language late and struggles with complex grammar, why is that?",
            "delayed language input deaf brain long term fluency",
            "is birth to five really the make or break stretch for deaf language?",
            "does waiting to see if speech works before ASL risk the language timing",
        ],
    },
    {
        "id": "asl_full_language",
        "canonical_question": "Stokoe proved sign language has its own building blocks like spoken words",
        "answer": "Yes, without qualification. American Sign Language is a full, natural human language. It is not a manual code for English and not English acted out on the hands. Its phonology, morphology, and syntax are entirely its own, borrowed from no spoken language, including English. As a Gallaudet graduate raised through the Maryland School for the Deaf, I can tell you ASL is a complete natural language with its own grammar. In 1960 William Stokoe proved this: he showed that signs are not undivided gestures or pantomime, because each one breaks into contrastive parts. He named three building-block parameters, handshape, location, and movement. Robbin Battison later added a fourth, palm orientation. ASL builds grammar in space and on the face, using signing space and nonmanual (facial) markers to convey questions, topics, and verb agreement, where spoken languages lean on word order or intonation. When Deaf children have full access, they acquire ASL on the very same developmental timeline that hearing children follow for spoken language. So labeling ASL as \"broken English\" or as pictures is factually wrong, and it is harmful. It is factually wrong and harmful because that lie is used to justify withholding sign language from Deaf children who need it. ASL is no fallback or last resort for people who cannot speak. It is a language in its own right, and for many of us it is the primary one we think, dream, and belong in.",
        "sources": [
            "Stokoe 1960 - Sign Language Structure",
            "Klima & Bellugi - The Signs of Language",
            "Sandler & Lillo-Martin - Sign Language and Linguistic Universals",
        ],
        "keyword_variants": [
            "in the 1960s William Stokoe showed ASL has its own building blocks the way spoken words do",
            "who proved sign language is structured like spoken language and put ASL on the same footing as English",
            "Stokoe 1960 Sign Language Structure duality of patterning cheremes",
            "handshape location movement are the contrastive building blocks of signs like phonemes",
            "ASL is a full natural language with its own phonology grammar not gesture or miming",
            "does signing have its own sublexical structure or is it just English on the hands",
            "the professor who put ASL on par with English as a real language",
            "1965 Dictionary of American Sign Language on Linguistic Principles Stokoe Croneberg Casterline",
            "Klima and Bellugi Signs of Language ASL linguistic universals",
            "ASL grammar syntax non-manual markers classifier predicates",
            "is ASL a real language or just broken English acted out on the hands",
            "sign language phonology minimal pairs contrastive parameters",
            "handshape location movement palm orientation fourth parameter Battison",
            "American Sign Language on the same footing as English",
            "ASL real language grammar",
            "is ASL a real language",
            "ASL grammar",
            "sign language linguistics",
            "Stokoe handshape",
            "ASL phonology",
            "ASL vs English",
            "is signing just gesture",
            "My mother-in-law keeps insisting that what Deaf people do with their hands is basically English acted out, so it doesn't really count as a language of its own. Is she right?",
            "I just got a Deaf student who signs, and I earnestly want to understand what linguists have actually established that makes ASL a complete language on par with spoken ones rather than a simplified communication system. Can you walk me through it?",
            "is American Sign Language a complete natural language",
            "ASL syntax morphology sentence structure",
            "someone told me sign language is just miming, is that actually true?",
            "does ASL have its own grammar or is it just broken English on the hands?",
            "my relative insists signing is only gestures and not a real language, how do I respond?",
            "handshape location movement contrastive sign parameters",
            "how do linguists show ASL qualifies as a genuine language",
            "is ASL its own separate language from spoken English",
            "sign language nonmanual markers facial grammar",
            "asl a real langauge or not",
            "prove ASL is a full natural human language not gesture",
            "Stokoe 1960 sign language structure phonology decomposition",
            "why do researchers say ASL is a real language with its own rules",
            "does signing count as an actual language with grammar",
            "how does ASL use space and the face to carry meaning",
            "as a teacher how do I explain ASL is not just English acted out",
            "palm orientation fourth parameter Battison sign phonology",
            "the myth that sign language is a set of pictures versus a real language",
        ],
    },
    {
        "id": "bilingual_bimodal",
        "canonical_question": "additive bilingualism deaf child two languages two modalities",
        "answer": "Bilingual-bimodal education gives a deaf child two languages in two modalities: ASL received visually, through the eyes, and English through print, with speech added where it genuinely helps. The two languages are developed together in an additive way, side by side, never in competition where one is withheld to force the other. A fully accessible first language, ASL taken in through the eyes, is the foundation on which a second language is built. That solid, visually accessible first language becomes the platform from which English reading and writing develop, so ASL is the bridge into English print literacy, not an obstacle to it. ASL is a full natural language with its own grammar, not a broken or word-for-word version of English. As Wyatte Hall makes clear, it is the lack of early accessible language, not deafness itself, that causes language deprivation and the later delays that follow. Deafness does not cause language deprivation; the lack of early accessible language does. Bilingualism is additive: a strong first language accelerates a second, so ASL and English reinforce each other instead of splitting a child's effort. Early access to sign supports reading growth and a secure Deaf identity. Decades of oral-only schooling that suppressed sign left many deaf adults under-literate in both English and ASL. When ASL is present from early childhood, both reading outcomes and Deaf identity are stronger. At the Maryland School for the Deaf, and at Gallaudet, I saw two languages taught alongside each other in one classroom, each strengthening the other, never a forced either-or choice.",
        "sources": [
            "Grosjean - The Right of the Deaf Child to Grow Up Bilingual",
            "Humphries et al 2012 - Language Acquisition for Deaf Children",
            "ThesisVault - Bilingual Bimodal Education",
        ],
        "keyword_variants": [
            "additive bilingualism two languages two modalities",
            "Grosjean right of the deaf child to grow up bilingual",
            "ASL English bilingual bimodal deaf education",
            "Petitto VL2 visual sign phonology and reading",
            "first-language ASL scaffold for English print literacy",
            "bi-bi bilingual bicultural deaf schooling",
            "does adding ASL improve deaf reading outcomes",
            "simultaneous sign and print exposure from infancy",
            "oralism LSL versus bilingual bimodal approach",
            "teaching a signed and a written language together in one classroom",
            "ASL as the bridge into English print literacy",
            "additive not subtractive bilingualism for deaf children",
            "bilingual bimodal deaf education ASL English",
            "bilingual bimodal",
            "ASL and English education",
            "deaf bilingual schooling",
            "why teach ASL and English",
            "deaf literacy through ASL",
            "additive bilingualism deaf",
            "oralism vs bilingual",
            "My baby was just identified as deaf and her pediatrician says we should pour everything into speech therapy and worry about reading later. Is there any real harm in raising her with sign language and written English side by side from the very start instead?",
            "Isn't teaching a deaf kid ASL and English at the same time just doubling the workload and setting both languages up to lag? Why do Deaf schools insist on running the two together rather than committing to one?",
            "teaching deaf children ASL and English at the same time",
            "should my deaf child learn sign and reading side by side?",
            "bimodal bilingual approach in a deaf classroom",
            "how do deaf kids learn to read English through ASL",
            "wont teaching my kid sign language hurt their English reading?",
            "ASL as the bridge into English literacy for deaf students",
            "why is bilingual schooling better than oral only for deaf kids",
            "two languages two modalities model for deaf education",
            "bilingual bimodel deaf education",
            "how do you teach a signed language and a written language in one room",
            "does adding ASL actually improve deaf reading outcomes",
            "strong first language sign then English second deaf approach",
            "as a policymaker which deaf education model does the evidence back?",
            "why did oral only schooling leave so many deaf adults under literate",
            "ASL and printed English taught together for deaf literacy",
            "can a deaf child become fully bilingual in ASL and English",
            "is it better to push speech only or teach sign and English both",
        ],
    },
    {
        "id": "deaf_gain",
        "canonical_question": "Deaf gain: what the wider world gains from a signing community, deafness as contribution not deficit",
        "answer": "Deaf gain flips the whole frame. Instead of asking what a deaf person lacks against a hearing norm, it asks what humanity gains from Deaf ways of being. The Deaf performance artist Aaron Williamson first voiced the phrase, and H-Dirksen Bauman and Joseph Murray developed and theorized it, intentionally inverting the deficit language of hearing loss into a language of contribution. What do Deaf communities contribute? A full, natural visual-spatial language, sign language, not a broken version of speech. We develop distinctive ways of organizing space, directing visual attention, and building community that most hearing people never build. We create a rich body of Deaf cultural production, including art, poetry, and film made in sign language. Deaf gain treats variation in human sensory and linguistic experience as a resource and an asset, not a defect to be corrected. Its core claim is simple: diversity in human bodies and languages benefits everyone rather than signaling a deficit. This builds on Carol Padden and Tom Humphries, whose work on Deaf culture established Deaf people as a cultural-linguistic minority with shared language, values, and history, like any people. As a Gallaudet graduate raised through Maryland School for the Deaf, I have lived this. So when you meet a deaf child as a member of a culture rather than a diagnosis, the deficit framing simply dissolves. You are not looking at loss. You are looking at a whole way of being human that has something to give.",
        "sources": [
            "Bauman & Murray - Deaf Gain: Raising the Stakes for Human Diversity",
            "Padden & Humphries - Deaf in America: Voices from a Culture",
            "Padden & Humphries - Inside Deaf Culture",
        ],
        "keyword_variants": [
            "a signing community contributes something the wider world benefits from, not just accommodated",
            "different perspectives make the whole stronger: the Deaf contribution to human diversity",
            "reframing deafness as a benefit and contribution rather than a deficit to accommodate",
            "Bauman and Murray, Deaf Gain: raising the stakes for human diversity",
            "Aaron Williamson coined deaf gain, inverting hearing loss into gain",
            "what humanity gains from Deaf ways of being, a diversity dividend",
            "deafness as a resource the wider world benefits from, more than accommodation",
            "deaf gain versus the deficit model of deafness",
            "why scholars argue deafness enriches human diversity",
            "Deaf Gain: Raising the Stakes for Human Diversity, Minnesota Press",
            "deaf not a deficit but a generative contribution to humanity",
            "does being Deaf add value the hearing world can benefit from",
            "deaf gain concept meaning and origin explained",
            "Deaf gain hearing loss",
            "Deaf gain",
            "hearing loss vs Deaf gain",
            "Bauman Murray",
            "Deaf culture benefit",
            "deaf not a deficit",
            "cultural linguistic minority deaf",
            "human diversity deaf",
            "The audiologist keeps describing my toddler's deafness only as a loss we need to correct, and it makes me uneasy. Is there a serious way of thinking that treats growing up deaf as adding something valuable to a child's life rather than just a shortfall?",
            "A friend mentioned there's a phrase first spoken by a Deaf performance artist and later developed by two scholars that deliberately turns the whole idea of 'hearing loss' upside down. Who originated it and what is it actually claiming?",
            "what do deaf people contribute to the world",
            "deaf gain concept meaning explained",
            "reframing deafness as a benefit instead of a loss",
            "how can being deaf actually be an advantage?",
            "isn't deafness just a disability, so what is this deaf gain idea about?",
            "Aaron Williamson origin of the deaf gain phrase",
            "visual spatial contributions of Deaf communities",
            "why do some scholars argue deafness enriches human diversity",
            "seeing deafness as a resource not a lack",
            "deffgain in deaf studies",
            "how does deaf gain push back on the medical view of deafness",
            "what does the world actually gain from Deaf ways of being",
            "Padden Humphries deaf as a cultural minority not patients",
            "are Deaf people a cultural group rather than people to be cured",
            "benefits of the Deaf perspective for human sensory diversity",
            "as a teacher how do I frame deafness positively for my class",
            "Deaf art poetry film and signed literature contributions",
        ],
    },
    {
        "id": "crpd_language_rights",
        "canonical_question": "CRPD treaty articles naming sign language as a human right",
        "answer": "When people ask me whether a deaf child has a right to sign language, I point to binding international law: the United Nations Convention on the Rights of Persons with Disabilities, the CRPD, which many countries have ratified as a treaty. It treats sign language as a language, plainly and repeatedly. Article 2 defines \"language\" to expressly include signed languages, so they cannot be ranked below spoken ones. Signed languages are full, natural human languages, equal in status to spoken languages, not gestures or a lesser code. Article 21 protects freedom of expression, including the right to use sign language in official interactions. Article 24 addresses education and calls for instruction delivered in sign language, for Deaf adult role models, and for environments that support the child's own language and identity. Article 30 recognizes Deaf people's distinct cultural and linguistic identity, including Deaf culture. Read together, these articles establish that a deaf child's access to sign language is a human right, not a service to be rationed or cut for budget reasons. Maartje De Meulder's scholarship documents how countries move from CRPD principles toward genuine legal recognition of sign languages, mapping the distinct types or degrees of legal recognition, from a bare constitutional mention to comprehensive language-rights legislation. As a Gallaudet alum and a proud product of the Maryland School for the Deaf, I lean on Maartje De Meulder's work to show that CRPD principles are meant to become real, enforceable legal recognition of sign languages, not empty promises.",
        "sources": [
            "CRPD/Article 2 - Definitions",
            "CRPD/Article 24 - Education",
            "CRPD/Article 30 - Cultural and Linguistic Identity",
        ],
        "keyword_variants": [
            "CRPD Articles 2 21 24 30 sign language",
            "UN Convention on the Rights of Persons with Disabilities treaty",
            "Article 2 defines sign languages as a language CRPD",
            "Article 21 freedom of expression in sign language CRPD",
            "Article 24 education in sign language General Comment 4",
            "Article 30 cultural and linguistic identity Deaf culture treaty",
            "Article 30(4) linguistic identity sign languages CRPD",
            "sign language as a human right under the UN CRPD",
            "World Federation of the Deaf CRPD interpretation",
            "Sign Language Peoples UNCRPD framing of Deaf communities",
            "UN CRPD Committee Concluding Observations sign language",
            "binding international human rights treaty sign language provisions",
            "US signed but not ratified CRPD treaty",
            "which CRPD articles back a sign language advocacy brief",
            "UNCRPD provisions on Deaf expression education cultural identity",
            "CRPD sign language deaf rights",
            "CRPD sign language",
            "UN convention deaf rights",
            "Article 24 education deaf",
            "sign language human right",
            "CRPD Article 2 21 30",
            "language rights deaf",
            "convention disabilities sign",
            "A relative keeps insisting that treating sign language as a 'real language' with equal standing is just feel-good political correctness with no legal weight behind it. Is there actually a binding international agreement that backs the idea that signed languages count as full languages?",
            "My daughter is deaf and her school keeps framing signing as an optional therapy service they can cut when budgets are tight. Does any human rights treaty actually say she has a right to be educated in a signed language rather than it being a favor they choose to grant?",
            "does international law protect the right to sign language",
            "which CRPD articles deal with deaf education and sign",
            "is access to sign language a human right under the UN?",
            "what does the disability convention actually say about sign language",
            "CRPD Article 21 freedom of expression in sign language",
            "how does the UN treaty define language for deaf people",
            "convention on the rights of persons with disabilities and Deaf identity",
            "does my deaf child have a legal right to a sign language education?",
            "CRPD treaty obligations on deaf rights",
            "CRDP sign language articles",
            "which CRPD articles cover Deaf culture and linguistic identity",
            "right to education in sign language international treaty",
            "De Meulder CRPD pathway to legal recognition of sign languages",
            "is signing recognized as a language by the United Nations",
            "what human rights protect a deaf child's access to sign language",
            "Article 30 Deaf cultural and linguistic identity treaty",
            "as a policymaker what does the CRPD require for sign language?",
            "UN disability convention deaf role models and sign education",
        ],
    },
    {
        "id": "legal_recognition",
        "canonical_question": "why two countries' sign language recognition laws differ in what Deaf people actually get",
        "answer": "When I talk about legal recognition, I mean far more than a law naming my sign language and calling the work done. As a Gallaudet alum who grew up at the Maryland School for the Deaf, I lean on Maartje De Meulder, the scholar who mapped out these different categories of sign language legal recognition. Maartje De Meulder distinguishes several distinct types or degrees of legal recognition, ranging from a symbolic constitutional mention at one end to comprehensive language-rights legislation at the other. Because Maartje De Meulder is the scholar who mapped this terrain, we can see that naming a language is only the lowest level, a constitutional mention that does not by itself guarantee Deaf people's needs are met. A country can write our sign language into its statutes and still fail us if nothing funds interpreters, bilingual schools, or Deaf teachers. Meaningful recognition attaches actual rights and resources: access to education, interpreting, and public services in sign, not words alone. De Meulder warns that when recognition is framed only as a disability accommodation, rather than as a full language right, our access stays fragile and easy to defund or cut. Framing carries concrete consequences, because it shapes the policy and money that follow. Recognized as a minority and community language, we gain standing as a linguistic and cultural community. Filed under disability services alone, that access stays precarious. That is why the full spectrum, from constitutional mention to comprehensive language-rights legislation, matters so much to me.",
        "sources": [
            "De Meulder 2015 - The Legal Recognition of Sign Languages",
            "De Meulder & Murray - Buttering Their Bread on Both Sides",
            "ThesisVault - Sign Language Recognition Typology",
        ],
        "keyword_variants": [
            "De Meulder typology of sign language recognition categories",
            "constitutional recognition versus statutory language-act recognition",
            "disability-framework recognition is De Meulder's weakest category",
            "symbolic recognition versus substantive linguistic rights",
            "the recognition-aspiration gap in Deaf communities",
            "Buttering Their Bread on Both Sides, De Meulder and Murray",
            "why merely naming a sign language in law delivers no downstream rights",
            "comparative-law survey of national sign language recognition regimes",
            "national statutory recognition versus a bare constitutional mention",
            "US ASL recognition is disability-framework only, with no federal statute",
            "what substantive provisions a national recognition act must contain",
            "two neighboring countries recognize their sign language but with different results",
            "how recognition is written into statute explains the gap between states",
            "recognition campaigns win symbolic naming but not the substantive aspirations",
            "which recognition category a country's sign language law falls into",
            "L1 education and interpreter provision that real recognition should deliver",
            "legal recognition sign languages",
            "legal recognition sign language",
            "De Meulder recognition",
            "sign language law",
            "language rights vs disability accommodation",
            "national sign language status",
            "recognizing ASL legally",
            "sign language legislation",
            "Our government just passed a law that officially names our national sign language, so does that mean Deaf people in our country now have everything they need?",
            "I teach Deaf students and I'm trying to understand why scholars insist there's a real difference between a country simply mentioning sign language in its constitution and a country that truly protects the rights of its Deaf community. Can you explain what's at stake in that distinction?",
            "how are sign languages recognized in the law",
            "different types of legal recognition for sign languages",
            "is ASL officially recognized as a language in US law?",
            "what does it take for a country to truly recognize a sign language?",
            "constitutional mention versus full sign language legislation",
            "does just naming a sign language in law actually help deaf people?",
            "sign language as a minority language versus a disability accommodation",
            "De Meulder typology of sign language recognition",
            "what national sign language legal status really means",
            "leagal recognition of sign language",
            "why does recognizing sign as a language matter for funding and services",
            "does legal recognition guarantee interpreters and Deaf teachers",
            "symbolic versus substantive sign language recognition",
            "how do countries grant sign language rights in practice",
            "recognizing a sign language as a genuine minority language",
            "as a lawmaker what should sign language legislation actually include",
            "difference between token recognition and real sign language rights",
            "sign language recognition tied to access to public services",
        ],
    },
    {
        "id": "cochlear_implant_myth",
        "canonical_question": "cochlear implant does not guarantee spoken language or replace signing",
        "answer": "No. As a Deaf person shaped by Gallaudet and the Maryland School for the Deaf, let me be plain. A cochlear implant can provide access to sound, but a device is not itself a language, and hearing sound is not the same as understanding language. Outcomes after implantation vary widely, and strong spoken language is never guaranteed. There is no sound evidence that ASL harms spoken-language development; early sign gives a language base that can support, not hinder, later speech. Sign language is the guaranteed channel: it gives your child full access to language right now, while the implant pathway of surgery, activation, mapping, and therapy develops on its own uncertain timeline. The belief that signing damages speech is an audist myth, not a research-supported finding. ASL is a complete natural language with its own grammar and phonology, not broken or simplified English.\n\nImplants and sign language are complementary rather than opposites, so a child can benefit from both at once. Research on bilingual-bimodal children, who grow up with both a signed and a spoken language, finds that implanted children raised with ASL do as well as or better than implant-only children on spoken-language outcomes, so adding sign takes nothing away from the implant. Providing ASL from birth guarantees a full language foundation regardless of how the implant performs. Early accessible language protects against language deprivation during the sensitive period for language acquisition.\n\nWyatte Hall and colleagues argue that withholding ASL to protect implant outcomes gambles with a child's one critical window for language. Wyatte Hall's scholarship frames delayed access to accessible language as a preventable public-health harm. Betting only on the implant risks a child ending up with neither fluent speech nor ASL if accessible input is delayed past the critical language-acquisition window. Give a child ASL now, and the implant becomes a welcome addition, not a gamble.",
        "sources": [
            "Hall, Hall & Caselli 2019 - Deaf Children Need Language, Not (Just) Speech",
            "Humphries et al 2012 - Language Choices for Deaf Children",
            "ThesisVault - Cochlear Implant Myths",
        ],
        "keyword_variants": [
            "surgeon said the implant takes care of his deafness",
            "does a cochlear implant settle how he will learn language",
            "cochlear implant is a device not a language",
            "CI gives auditory access but sound is not language",
            "spoken language after implantation is never guaranteed, outcomes vary",
            "myth that an implant removes the need to sign",
            "no evidence that early sign harms spoken language in CI users",
            "audiologist says stop signing after activation",
            "LSL listening and spoken language only pathway risk",
            "CI and sign are parallel not alternative, bilingual bimodal",
            "AG Bell oralism claim that sign delays speech",
            "Hall Hall Caselli deaf children need language not just speech",
            "implanted child still needs a visual first language",
            "cochlear implant does not restore normal hearing",
            "cochlear implant ASL language deaf child",
            "cochlear implant language",
            "does CI replace ASL",
            "implant myth deaf",
            "CI vs sign language",
            "cochlear implant outcomes",
            "do implanted kids need ASL",
            "implant fix deafness",
            "My baby was just approved for a cochlear implant, so is there really any point in also raising her with sign language?",
            "A colleague insists that letting deaf children sign will hold back their speech once they're implanted, so is that claim actually supported by research?",
            "cochlear implant enough",
            "will an implant fix my baby's hearing so we can skip sign",
            "my pediatrician says the CI is all he needs, is that true",
            "does a cochlear implant mean my kid can learn to talk like normal",
            "implanted child still needs sign language?",
            "CI kid speech only no ASL risk",
            "my brother got his deaf son implanted and refuses to sign, should I worry",
            "is it true once you get the implant you dont need sign language anymore",
            "cochlear implant vs learning ASL for toddler",
            "as a teacher I have implanted students who still struggle in language, why",
            "cochlear implant garuntee spoken language kid",
            "do deaf kids with implants fall behind in language development",
            "should we bet everything on the implant working for our deaf newborn",
            "policy question does implantation remove the need for ASL services",
            "hearing aids and cochlear implants replace sign language right",
            "implant surgery deaf baby language foundation",
            "my audiologist told us no signing after the implant, is that harmful",
            "cant the implant just cure his deafness completely",
        ],
    },
    {
        "id": "gallaudet",
        "canonical_question": "Gallaudet University deaf, Washington DC",
        "answer": "Gallaudet University, located in Washington, D.C., is the only university in the world built from its very start for deaf and hard-of-hearing students. It was established in 1864 through a federal charter, and it is designed specifically around deaf and hard-of-hearing students. ASL and English are the languages of teaching and of daily campus life, so access is designed in from the outset rather than something a student must request. On this campus being Deaf is simply the norm, and accommodation is built in, never treated as a special exception. As a Gallaudet alum, I know its real gift. Far from isolating anyone, it gives many students their first setting of complete language access, where communication flows without barriers and language surrounds them freely and fully. That environment supports language development instead of depriving it (the harm Wyatte Hall documents in language deprivation), and a fully accessible language environment strengthens rather than weakens readiness for adult and professional life. Gallaudet produces Deaf scholars, educators, researchers, and leaders who sustain our community's knowledge, and its graduates go on to become teachers, leaders, researchers, and professionals in every field who carry that knowledge forward. The 1988 Deaf President Now protest, in which students halted the campus until a Deaf president, I. King Jordan, was appointed the university's first, remains a landmark in Deaf civil rights and self-determination. Gallaudet shows what becomes possible when a whole institution is built around us.",
        "sources": [
            "Gallaudet University - Deaf President Now 1988",
            "Padden & Humphries - Inside Deaf Culture",
            "ThesisVault - Gallaudet History and DPN",
        ],
        "keyword_variants": [
            "Gallaudet University",
            "deaf university in Washington DC",
            "Deaf President Now 1988, first Deaf president I King Jordan",
            "students shut down Gallaudet campus in the 1988 DPN protest",
            "Gallaudet deaf college in Washington DC",
            "what is Gallaudet",
            "only deaf college",
            "My daughter is Deaf, and her IEP team keeps mentioning a school in Washington, D.C. where every class is taught in sign language. What is that place and why would it be a good fit for her?",
            "why Deaf President Now still matters, Deaf civil rights landmark",
            "who became the first deaf president of Gallaudet",
            "Gallaudet federal charter Lincoln 1864 history",
            "tell me about the Deaf President Now movement",
            "Gallaudet University deaf",
            "deaf university",
            "Deaf President Now",
            "DPN 1988",
            "Gallaudet ASL campus",
            "Honestly, doesn't sending a deaf teenager to a college only for deaf people just wall them off from the hearing world instead of preparing them to live in it?",
            "that all deaf university in DC",
            "whats the deal with the college where every student is deaf",
            "deaf students protest for a deaf president 1988",
            "as a hearing parent should my deaf teen apply to Gallaudet",
            "why is Gallaudet so important to the deaf community",
            "the deaf college where everyone signs ASL",
            "Galudet university washington dc deaf",
            "grad school question what makes Gallaudet a language rich campus",
            "is there really a whole university taught entirely in sign language",
            "Gallaudet deaf civil rights landmark protest story",
            "my niece got into Gallaudet what is campus life like there",
            "washington dc deaf university sign language instruction",
            "DPN protest students shut down the campus",
            "federal funding for the national deaf university Gallaudet",
            "Gallaudet alumni deaf leaders scholars teachers",
        ],
    },
    {
        "id": "maryland_school_deaf",
        "canonical_question": "Maryland School for the Deaf helps a late-signing child with language gaps",
        "answer": "The Maryland School for the Deaf runs two campuses, one in Frederick and one in Columbia, and on both ASL and English live side by side from the earliest grades. ASL is the primary language of instruction, not an accommodation, so a deaf child gets lessons, discussions, and all the incidental learning of a school day directly, without waiting on an interpreter. For a child who arrives late to sign, already carrying language gaps, that immersion is the remediation itself: surrounded by fluent Deaf teachers and signing peers, the child can still acquire a full first language. Deaf alumni return as Deaf teachers, and language and culture pass from Deaf staff and fluent signing peers to the young, an intergenerational thread that runs back through generations of alumni. MSD cites the national statistic that more than 95 percent of deaf and hard of hearing children are born to hearing families who do not sign; that is a nationwide figure, not a count of MSD's own families, and it is why the school's Family Education program teaches families ASL free of charge during the birth to five years, when brain development is at its most active. Governance now matches the mission: under House Bill 1052, Chapter 721, effective July 1, 2026, the Board of Trustees has thirteen members, at least seven of them deaf. A November 2025 Baltimore Sun guest commentary by Kelby Brick, Wyatte Hall, and Superintendent John Serrano calls MSD a safe harbor for these children. That is the quiet power of a school where every hand speaks.",
        "keyword_variants": [
            "relocating to Maryland, our late-signing seven-year-old has big language gaps",
            "everyone steers us toward the Deaf school in Frederick over the local district",
            "what specifically about that school helps a child who arrived with so little language",
            "how does MSD help a deaf kid who showed up with almost no language finally catch up",
            "Maryland School for the Deaf Frederick and Columbia campuses",
            "MSD residential K-12 School for the Deaf, everyone signs all day",
            "IEP placement: signing School for the Deaf versus local mainstream district",
            "mainstream with an interpreter versus a School for the Deaf where teachers sign",
            "illusion of inclusion, mainstreaming alone does not prevent language deprivation",
            "IDEA Part B least restrictive environment misread for a language-deprived deaf child",
            "signing environment with Deaf teachers, Deaf peers, and Deaf staff",
            "language-rich School for the Deaf lets a late-exposed child catch up",
            "is the state Deaf school in Frederick better than the mainstream program",
            "residential day program placement for a deaf child with little first language",
            "Maryland School for the Deaf",
            "MSD Frederick",
            "deaf school bilingual",
            "language rich deaf school",
            "residential deaf school",
            "MSD ASL English",
            "deaf boarding school Maryland",
            "My son is deaf and everyone keeps telling me the state school up in Frederick would be better for his language than the mainstream program he's in now, but honestly, wouldn't surrounding him with only other deaf kids just hold him back?",
            "I teach elementary students and I'm trying to understand the actual mechanism: what happens at MSD's Frederick and Columbia campuses that lets a deaf kid who showed up with almost no language finally start catching up?",
            "MSD Columbia Maryland deaf campus",
            "best deaf school in Maryland for my son",
            "does the Maryland deaf school teach kids in sign",
            "Frederick Maryland school for deaf children ASL",
            "my daughter is deaf is a residential school in Maryland good",
            "state school for the deaf Maryland worth it",
            "Merryland school for the deaf frederick",
            "how does MSD help kids who arrive with no language",
            "teacher here curious how MSD runs bilingual instruction",
            "deaf kids thrive at the Maryland school signing environment",
            "is MSD a good place for language deprived deaf children",
            "Maryland deaf school Deaf culture and deaf staff",
            "should I send my deaf child to the Frederick campus",
            "grad student researching the Maryland School for the Deaf bilingual model",
            "public deaf school Maryland ASL and English in one classroom",
            "MSD deaf mentors older students pass down culture",
            "whats a language rich school for deaf children in MD",
            "Maryland school deaf full communication access every teacher signs"
        ],
        "sources": [
            "Maryland School for the Deaf",
            "Bilingual Bimodal Protective Factors",
            "Critical Period and Brain",
            "American Sign Language"
        ]
    },
    {
        "id": "deaf_community_transmission",
        "canonical_question": "how ASL survives and is handed down across generations when almost no deaf children have signing parents",
        "answer": "Most languages pass down parent-to-child in the home, but ASL usually does not, and the reason is simple: roughly ninety-five percent of deaf children are born to hearing parents who do not sign at birth. Because about ninety-five percent of us have hearing parents who initially do not sign, the home is usually not where a deaf child first meets fluent ASL. That matters, because a deaf child needs early access to a fully accessible visual language to develop language on a typical timeline. Deafness itself never causes delay; the absence of reachable input does. Since families often cannot yet supply fluent signed input, the Deaf community becomes the main channel that carries ASL forward. The community itself functions as the transmission line for the language across generations. Deaf schools, Deaf clubs, Deaf mentors, and Deaf peers transmit ASL and Deaf culture, frequently more than a child's own family does. Deaf peers and mentors are central to how children acquire both ASL and Deaf cultural belonging. Scholars such as Ben Bahan describe this as horizontal transmission: it flows from Deaf adults and older children to younger ones, not only vertically from parent to child, a pattern Bahan and others have documented. Residential and bilingual Deaf schools are where many of us first encounter fluent language and a cultural home. Because our language depends on shared Deaf spaces rather than family inheritance, the community is uniquely fragile, and protecting those spaces protects ASL itself. The community raises its own, and I am one of them.",
        "sources": [
            "Bahan - Face-to-Face Tradition in the American Deaf Community",
            "Padden & Humphries - Deaf in America",
            "ThesisVault - Cultural Transmission of ASL",
        ],
        "keyword_variants": [
            "how is ASL passed down when so few deaf kids have signing parents",
            "how has ASL survived generation to generation",
            "sign language handed down through Deaf schools and peers not parents",
            "horizontal transmission of ASL among Deaf children at school",
            "residential schools for the deaf as the cradle of ASL transmission",
            "Deaf clubs and Deaf mentors carry the language to the next generation",
            "older Deaf students and Deaf staff pass signing to younger children",
            "Padden and Humphries Deaf in America, Deaf as a people not a patient population",
            "Ben Bahan Face-to-Face Tradition ASL storytelling lineage",
            "ninety-five percent of deaf children are born to hearing parents",
            "why deaf kids learn ASL from the community not from home",
            "cultural transmission of ASL and Deaf identity outside the family",
            "why residential deaf schools are vital to the survival of ASL",
            "the Deaf community itself hands its language and heritage forward",
            "Deaf culture ASL transmission generation",
            "how is ASL passed down",
            "deaf culture transmission",
            "95 percent hearing parents",
            "deaf community language",
            "Ben Bahan tradition",
            "who teaches deaf kids ASL",
            "horizontal language transmission",
            "My newborn is deaf and no one in our family signs. Won't she just soak up ASL from us at home, the same way a hearing baby picks up whatever language the parents speak?",
            "I just started teaching in a Deaf education program and I'm surprised how much the older Deaf students and the Deaf staff seem to shape the little ones' language and identity. Why does so much of that come from the community instead of from the kids' own homes?",
            "why do deaf kids learn sign from other deaf people not their parents",
            "most deaf children are born to hearing parents statistic",
            "how does Deaf culture get passed to the next generation",
            "deaf kids pick up ASL at school not at home why is that",
            "who actually teaches a deaf baby to sign if the parents cant",
            "ninety five percent of deaf kids born to hearing families",
            "grad student horizontal versus vertical transmission of sign language",
            "deaf schools deaf clubs deaf mentors pass down the language",
            "as a hearing mom how will my deaf child learn Deaf culture",
            "why are residential deaf schools so vital for cultural survival",
            "deaf peers and older students carry ASL forward",
            "sign language not passed from parent to child explanation",
            "is the deaf community fragile because of how the language spreads",
            "how is deaf identity handed down between generations",
            "Ben Bahan face to face tradition American deaf community",
            "deaf child hearing parents dont sign what happens to the culture",
            "why the community itself raises deaf kids language wise",
            "cultural transmission of ASL and the role of residential school",
        ],
    },
    {
        "id": "interpreting_access",
        "canonical_question": "qualified interpreter is not full communication access",
        "answer": "Skilled, certified interpreters are essential, and they genuinely open doors that would otherwise stay shut. But booking one is not the same as full accessibility. Real access means a deaf person can participate directly and fully, not always working through a relay or a third party. It means accurate captioning and video relay that actually functions, not just an interpreter in the corner. It means Deaf peers, Deaf role models, and Deaf professionals in the room, and institutions and schools built with us from the start rather than retrofitted after the fact. In high-stakes settings like hospitals and courtrooms, an interpreter must be certified and neutral: leaning on a family member or a phone app there is a failure, not a shortcut. A signed conversation with a fluent peer is not the same as an interpreted one, and direct communication carries its own value, so a community of fluent signing peers is itself part of access. For a deaf child this matters most. Genuine access centers direct communication in the child's own language, not only information relayed through an interpreter. Full, accessible language exposure early is what protects against language deprivation, as Wyatte Hall and other scholars have documented. CRPD Article 24 supports education delivered in sign language, in environments with Deaf role models that maximize a deaf child's development. Alongside qualified, neutral interpreters, accurate captioning and reliable video relay round out real access. The goal is a world where our language belongs.",
        "sources": [
            "ThesisVault - Interpreting and Communication Access",
            "CRPD/Article 9 - Accessibility",
            "De Meulder - Sign Language Rights and Access",
        ],
        "keyword_variants": [
            "is it acceptable to ask a Deaf patient's adult son to interpret in a clinic",
            "using a video interpreting app for a Deaf patient in a busy clinic",
            "what real communication access requires in a medical setting for a Deaf patient",
            "why a family member or adult child must not interpret for a Deaf patient",
            "booking a sign language interpreter for every meeting does not make the workplace fully accessible to a Deaf employee",
            "genuine communication access for a Deaf student beyond assigning someone to sign",
            "qualified certified interpreter in a hospital court or workplace",
            "ADA effective communication auxiliary aids for Deaf patients and employees",
            "video remote interpreting VRI and CART captioning in high-stakes settings",
            "De Meulder interpreting services a quick fix for inclusion",
            "certified neutral interpreter required in court and legal proceedings",
            "informed consent hospital interpreter CRPD Article 21",
            "direct communication access versus relay through a third party",
            "Title III healthcare provider must provide a qualified interpreter",
            "children should never interpret in clinical or legal high-stakes settings",
            "deaf access beyond interpreter",
            "deaf access interpreter",
            "more than an interpreter",
            "qualified interpreter",
            "video relay captioning",
            "communication access deaf",
            "direct access ASL",
            "interpreter hospital court",
            "My company always books a sign language interpreter whenever our deaf employee has a meeting, so isn't the workplace already fully accessible to him?",
            "My daughter is deaf and I want her school to truly do right by her, so what does genuine communication access look like beyond just assigning someone to sign for her?",
            "is booking an interpreter enough for deaf access",
            "why isnt an interpreter always enough for a deaf person",
            "deaf patient hospital communication rights certified interpreter",
            "can I just use a family member to interpret for my deaf relative",
            "what does real access look like for deaf employees at work",
            "certified neutral interpreter court legal deaf setting",
            "captions video relay direct communication for deaf people",
            "as an employer what access does my deaf worker need beyond an interpreter",
            "interpeter phone app deaf in the hospital okay?",
            "difference between an interpreted conversation and a direct signed one",
            "deaf accessibility is more than just translation",
            "why deaf people want institutions built with them not retrofitted",
            "using a phone app to interpret for a deaf person in an emergency",
            "communication access requirements deaf public services policy",
            "deaf professional in the room versus relay through an interpreter",
            "qualified interpreter high stakes deaf medical or legal setting",
            "is captioning accurate enough to count as deaf access",
            "my deaf coworker says interpreters arent everything what does he mean",
        ],
    },
    {
        "id": "deaf_identity_audism",
        "canonical_question": "Capital-D Deaf identity versus lowercase deaf; audism discrimination",
        "answer": "When I write deaf with a lowercase d, I mean audiological or hearing status, how a body hears. When I write Deaf with a capital D, I name a cultural and linguistic identity: belonging to a signing community with its own shared language, history, and values. So whether a person claims a Deaf identity does not depend on their degree of hearing loss or how much sound they can hear. Deaf identity is not an audiological measurement at all. It is a matter of culture and belonging. Deaf pride says we are whole persons who carry a full language and a living culture. Deaf gain says the world is richer and enriched by Deaf people being part of it.\n\nAudism is a real term, coined by Deaf scholar Tom Humphries. It is the assumption that being able to hear and speak makes a person more capable or more human. Audism also names the discrimination and unequal treatment that flow from that assumption. Concretely, it looks like withholding ASL, banning signing in classrooms, and treating a deaf child as a broken hearing child to be fixed. Denying a child access to sign language while framing deafness as a deficit to be fixed is what audism looks like, not support for that child's communication. Naming audism lets us recognize and see the bias clearly, plainly, instead of accepting it as normal.",
        "sources": [
            "Humphries - Deaf Culture and the Concept of Audism",
            "Padden & Humphries - Deaf in America",
            "Bauman - Audism: Exploring the Metaphysics of Oppression",
        ],
        "keyword_variants": [
            "capital D Deaf versus lowercase d deaf orthographic convention",
            "why is the D capitalized in Deaf",
            "Deaf capital D cultural linguistic identity not audiological status",
            "lowercase deaf means audiological hearing status not community membership",
            "audism definition prejudice that hearing and speaking are superior",
            "Tom Humphries coined the term audism in 1975",
            "banning sign language in schools as audism oralism",
            "Dirksen Bauman audism the metaphysics of oppression",
            "phonocentrism belief that speech makes one more human",
            "is suppressing signing a form of audist discrimination",
            "Padden Humphries Deaf in America cultural identity",
            "when to capitalize Deaf editorial capitalization rule",
            "audism systemic discrimination against Deaf people",
            "does forbidding students to sign count as audism",
            "Deaf identity does not depend on degree of hearing loss",
            "Deaf identity audism",
            "Deaf identity",
            "big D little d deaf",
            "what is audism",
            "deaf culture identity",
            "audism discrimination",
            "capital D Deaf meaning",
            "deaf pride",
            "Our audiologist keeps measuring my daughter's hearing levels, but a teacher of the deaf told us there's also a whole community-and-identity side to being deaf. How is belonging to that community actually different from just not being able to hear well?",
            "People keep using the word 'audism' and honestly it sounds like a made-up buzzword to me. Where did it even come from, and isn't wanting a deaf kid to learn to hear and talk just normal parenting rather than some kind of prejudice?",
            "what does capital D Deaf mean versus lowercase deaf",
            "audism definition discrimination against deaf people",
            "why do some deaf people capitalize the D in Deaf",
            "is thinking hearing is better than being deaf a kind of bias",
            "deaf as a culture not a disability explain",
            "audism who coined the term",
            "my relative thinks deaf kids just need fixing whats that attitude called",
            "deaf pride identity belonging signing community",
            "audizm against deaf and hard of hearing",
            "grad student audism as the metaphysics of oppression",
            "banning signing in classrooms as an example of audism",
            "difference between deaf hearing status and Deaf cultural identity",
            "is treating a deaf child as a broken hearing child discrimination",
            "what is the bias that says speaking makes you more human",
            "Tom Humphries and the concept of audism",
            "deaf identity shared language history and values",
            "why the deaf community sees itself as whole not broken",
            "audism in schools and the medical system toward deaf kids",
        ],
    },
    {
        "id": "deaf_world_visual",
        "canonical_question": "ASL poetry, use of space, DEAF-WORLD as a visual people",
        "answer": "We Deaf people describe ourselves as a visual people. Our sight organizes how we take in and structure experience, and it is a whole way of being, not a compensation or substitute for a missing sense. We are fundamentally visual in how we perceive and organize the world.\n\nThe ASL sign DEAF-WORLD names the Deaf community as a whole: a distinct cultural and linguistic group with its own norms, not merely people with a hearing loss. In \"A Journey into the DEAF-WORLD,\" scholars Ben Bahan, Harlan Lane, and Robert Hoffmeister describe this community and its visual life. In their scholarship, Ben Bahan, Harlan Lane, and Robert Hoffmeister theorize the DEAF-WORLD as a linguistic community.\n\nThat visual life shows itself everywhere: in ASL, a language built in space that reflects this visual way of being; in a storytelling tradition rich in classifiers and role-shift; and in social customs tuned to eye contact, light, and shared visual attention.\n\nScholars call this Deaf gain (Bauman and Murray developed the idea): presence and contribution, not deficit or loss. When people frame deafness only as silence or absence, they miss a world that is full, expressive, and complete.\n\nTo belong to the DEAF-WORLD is to be part of a people, not to lack a sense. If you are raising a deaf child, know this: your child is not missing something. Your child belongs to a visual people, a real culture and language, whole from the start.",
        "sources": [
            "Lane, Hoffmeister & Bahan - A Journey into the DEAF-WORLD",
            "Bahan - Upon the Formation of a Visual Variety of the Human Race",
            "Bauman & Murray - Deaf Gain",
        ],
        "keyword_variants": [
            "DEAF-WORLD visual variety of the human race Bahan",
            "ASL poetry use of space and perspective shift performance",
            "visual vernacular whole-body storytelling tradition",
            "Lane Hoffmeister Bahan A Journey into the DEAF-WORLD",
            "Deaf Gain Bauman Murray visual way of being",
            "classifier predicates and role shift in ASL narrative",
            "signing space depiction and constructed action",
            "Deaf people as a visual people experiencing the world through the eyes",
            "visual-spatial life-world of the Deaf community",
            "how vision organizes ASL grammar and Deaf artistic expression",
            "Ben Bahan Bird of a Different Feather visual storytelling",
            "sight-centered experience not compensation for hearing loss",
            "ASL literature poetry classifiers spatial narrative",
            "the Deaf-World as its own visual community",
            "Deaf world visual people",
            "DEAF-WORLD",
            "visual people deaf",
            "deaf visual language",
            "Journey into the Deaf World",
            "Bahan visual variety",
            "deaf community as a people",
            "why deaf are visual",
            "Some Deaf adults told me that they 'take in the world through their eyes' and belong to their own group with its own way of life, but honestly, isn't that just putting a positive spin on not being able to hear?",
            "I'm a teacher who just started working with a Deaf student, and I want to understand what people mean when they call the Deaf community its own people with an experience of life centered on sight, and how their language reflects that. Can you explain?",
            "what does the DEAF-WORLD sign actually mean",
            "are deaf people really a visual people",
            "deaf visual way of taking in and experiencing the world",
            "why do deaf folks say vision is a whole way of being not a compensation",
            "Harlan Lane Hoffmeister Bahan a journey into the deaf world book",
            "deaf storytelling classifiers role shift visual tradition",
            "grad student DEAF-WORLD as a distinct linguistic community",
            "is deafness just silence and absence or is it something fuller",
            "deaf culture eye contact light and shared attention customs",
            "visual variety of the human race what does that mean",
            "how vision shapes ASL grammar and deaf social customs",
            "deaf people belong to a people not to a missing sense",
            "deff world visual community explained",
            "what makes the deaf world its own distinct culture",
            "deaf gain as a visual experience in lived form",
            "why is sign language built in space and organized by vision",
            "the deaf community as its own whole world explained",
            "deaf people organize experience visually rather than by sound",
        ],
    },
]

# The banner voice, printed at the top of every human-readable run.
BANNER = (
    "I am a Deaf scholar, Gallaudet-raised and rooted at the Maryland School for the\n"
    "Deaf, and I hold one truth without wavering. Signed language is a full human\n"
    "language, and every deaf child is born with a right to it. I have watched language\n"
    "deprivation up close, and I have watched a child bloom the moment ASL fills the\n"
    "room. This brain answers from the thesis vault in that voice, so the science and\n"
    "the sources stay honest. We are not a deficit. We are Deaf gain."
)


# -- small helpers ------------------------------------------------------------
def safe_stats(cache):
    """Call cache.stats() defensively. A malformed result falls back to zeros."""
    try:
        s = cache.stats()
        if not isinstance(s, dict):
            raise ValueError("stats did not return a dict")
        return {
            "count": int(s.get("count", 0) or 0),
            "dir": s.get("dir", ""),
            "bytes": int(s.get("bytes", 0) or 0),
            "hits": int(s.get("hits", 0) or 0),
            "bank": float(s.get("bank", 0.0) or 0.0),
        }
    except Exception:
        return {"count": 0, "dir": "", "bytes": 0, "hits": 0, "bank": 0.0}


def estimate_principal(answer_text, has_sources):
    """Estimate the deposit store() will make (EVICTION RESISTANCE, not
    reachability) by running the library's own formula, so the --dry-run
    estimate always matches what store() actually deposits."""
    return _principal(answer_text, ["grounded"] if has_sources else [])


def word_count(answer_text):
    return len(_tokens(answer_text))


def truncate(text, width):
    text = str(text)
    if len(text) <= width:
        return text
    return text[: max(0, width - 3)] + "..."


# -- the prime / verify loop --------------------------------------------------
def prime(cache, concepts, dry_run, quiet):
    """Seed each concept for reachability, then drill its phrasings to compound
    them and verify.

    Reachability and eviction resistance are two separate properties (see the
    module docstring). This function establishes REACHABILITY by seeding: it
    stores the canonical answer under the short keyword-dense canonical question
    AND under every keyword variant, so each phrasing lands an exact normalized
    match in lookup(). It then drills each variant with a lookup, which compounds
    the matched entry (hits += 1, value x1.15, recency reset) for EVICTION
    RESISTANCE and verifies the phrasing is reachable. Drilling never creates
    reachability; only seeding does.

    In --dry-run this performs NO store() and NO lookup() (lookup mutates on a
    hit, so a true dry-run cannot call it) and only reports the plan.

    Returns a results dict the reporter consumes.
    """
    # Identity map: normalized phrasing -> concept id, covering the canonical
    # question AND every keyword variant, because each variant is now seeded as
    # its own canonical. This is how I detect whether a drilled variant resolved
    # to the right concept WITHOUT polluting the answer body with markers. The
    # brain would otherwise serve a marker to a real user, so the answer text
    # stays clean.
    norm_to_concept = {}
    for c in concepts:
        norm_to_concept[normalize(c["canonical_question"])] = c["id"]
        for v in c.get("keyword_variants", []):
            norm_to_concept[normalize(v)] = c["id"]

    results = {
        "concepts": [],          # per-concept detail rows
        "seeded": 0,             # concepts whose canonical store() succeeded
        "seed_failures": [],     # ids whose canonical store() returned None
        "misses": [],            # {concept, variant, keywords, note}
        "wrong": [],             # {variant, expected, got}
        "foreign": [],           # {concept, variant}
        "totals": {"HIT": 0, "MISS": 0, "WRONG": 0, "FOREIGN": 0, "ERROR": 0},
        "variants_total": 0,
        "compounding_events": 0,
    }

    for c in concepts:
        cid = c["id"]
        question = c["canonical_question"]
        answer = c["answer"]
        sources = c["sources"]
        variants = c.get("keyword_variants", [])
        w = word_count(answer)
        p_est = estimate_principal(answer, bool(sources))

        row = {
            "id": cid,
            "question": question,
            "answer_words": w,
            "principal": p_est,
            "variants_total": len(variants),
            "HIT": 0, "MISS": 0, "WRONG": 0, "FOREIGN": 0, "ERROR": 0,
            "seed_ok": None,      # True/False in live mode, None in dry-run
        }

        # -- DRY-RUN: report the plan only, no store and no lookup ----------
        if dry_run:
            if not quiet:
                print(
                    "  [plan] " + cid + ": seed 1 keyword-dense canonical ("
                    + str(w) + " words, est principal P=" + ("%.2f" % p_est)
                    + ") plus " + str(len(variants)) + " variant canonical(s) "
                    "for reachability, then drill " + str(len(variants))
                    + " phrasing(s) to compound and verify."
                )
            results["variants_total"] += len(variants)
            results["concepts"].append(row)
            continue

        # -- PHASE 1a: seed the keyword-dense canonical question ------------
        try:
            path = cache.store(question, answer, sources, DEFAULT_MODEL)
        except Exception as e:
            path = None
            if not quiet:
                print("  [seed] " + cid + ": canonical store raised (" + repr(e) + "), continuing.")

        if path:
            row["seed_ok"] = True
            results["seeded"] += 1
            if not quiet:
                print("  [seed] " + cid + ": stored canonical (" + str(w) + " words, P=" + ("%.2f" % p_est) + ").")
        else:
            row["seed_ok"] = False
            results["seed_failures"].append(cid)
            if not quiet:
                print("  [seed] " + cid + ": canonical store returned nothing, this canonical did not land.")

        # -- PHASE 1b: seed every variant as its own canonical --------------
        # Reachability is established HERE, by storing the same answer under each
        # phrasing so an exact normalized match reaches it. A variant whose store
        # fails will simply MISS in the drill below, which is a real defect the
        # report surfaces, not a normal outcome.
        for v in variants:
            try:
                cache.store(v, answer, sources, DEFAULT_MODEL)
            except Exception as e:
                if not quiet:
                    print("    [seed] variant \"" + v + "\": store raised (" + repr(e) + ").")

        # -- PHASE 2: drill each variant to compound and verify -------------
        for v in variants:
            results["variants_total"] += 1
            try:
                view = cache.lookup(v)
            except Exception as e:
                row["ERROR"] += 1
                results["totals"]["ERROR"] += 1
                if not quiet:
                    print("    [error] variant \"" + v + "\": lookup raised (" + repr(e) + ").")
                continue

            if view is None:
                # A seed that did not land: the phrasing is not reachable. This
                # is a defect to fix, not a training signal.
                row["MISS"] += 1
                results["totals"]["MISS"] += 1
                kws = sorted(content_words(v))
                negs = sorted(negation_words(v))
                note = (
                    "this phrasing is not reachable, so its seed did not land. Seed it "
                    "as its own canonical (store the same answer under this exact "
                    "phrasing) so an exact normalized match reaches it, or shorten the "
                    "CANONICAL question toward these keywords {" + ", ".join(kws)
                    + "} so this probe clears content-word Jaccard >= 0.70. The blocking "
                    "extra content words live in the canonical question, not in this "
                    "variant, so trimming the variant cannot help."
                )
                if negs:
                    note = ("negation {" + ", ".join(negs) + "} blocks the fuzzy scan "
                            "unless the canonical carries the same negation. " + note)
                results["misses"].append({
                    "concept": cid, "variant": v, "keywords": kws, "note": note,
                })
                if not quiet:
                    print("    [MISS] variant \"" + v + "\" did not resolve; reduced to {" + ", ".join(kws) + "}.")
                continue

            # A hit occurred, so the entry compounded (value x1.15, clock reset).
            # Compounding is EVICTION RESISTANCE only; it did not create the
            # reachability that produced this hit (seeding did).
            results["compounding_events"] += 1
            returned_norm = view.get("normalized") or normalize(view.get("question", ""))
            got_id = norm_to_concept.get(returned_norm)

            if got_id == cid:
                row["HIT"] += 1
                results["totals"]["HIT"] += 1
                if not quiet:
                    print("    [HIT] variant \"" + v + "\" resolved to " + cid + " and compounded it.")
            elif got_id is not None:
                # Cross-talk: matched a DIFFERENT seeded concept.
                row["WRONG"] += 1
                results["totals"]["WRONG"] += 1
                results["wrong"].append({"variant": v, "expected": cid, "got": got_id})
                if not quiet:
                    print("    [WRONG] variant \"" + v + "\" expected " + cid + " but hit " + got_id + ".")
            else:
                # Landed on a pre-existing entry that predates this run.
                row["FOREIGN"] += 1
                results["totals"]["FOREIGN"] += 1
                results["foreign"].append({"concept": cid, "variant": v})
                if not quiet:
                    print("    [FOREIGN] variant \"" + v + "\" hit an existing non-seed entry.")

        results["concepts"].append(row)

    return results


# -- reporting ----------------------------------------------------------------
def print_report(resolved_dir, mode, stats_before, stats_after, results, elapsed, exit_code):
    """Print the final human-readable report. Bank delta is the headline number."""
    line = "-" * 78
    print("")
    print(line)
    print("PRIME REPORT")
    print(line)
    print("cache_dir : " + resolved_dir)
    print("model     : " + DEFAULT_MODEL)
    print("vault     : " + VAULT_NAME)
    print("mode      : " + mode)
    print("")

    b, a = stats_before, stats_after
    print("KNOWLEDGE BANK, before and after:")
    print("  BEFORE:  count=%d  bytes=%d  hits=%d  bank=%.4f" % (b["count"], b["bytes"], b["hits"], b["bank"]))
    print("  AFTER :  count=%d  bytes=%d  hits=%d  bank=%.4f" % (a["count"], a["bytes"], a["hits"], a["bank"]))
    d_count = a["count"] - b["count"]
    d_bytes = a["bytes"] - b["bytes"]
    d_hits = a["hits"] - b["hits"]
    d_bank = a["bank"] - b["bank"]
    print("  DELTA :  count=%+d  bytes=%+d  hits=%+d  bank=%+.4f" % (d_count, d_bytes, d_hits, d_bank))
    print("  Bank delta is the proof: %+.4f of eviction-resistant value deposited by this run." % d_bank)
    if d_bytes < 0:
        print("  Note: bytes fell, which means a store triggered eviction. That is healthy, not a loss.")
    print("")

    # Per-concept table, weakest coverage first so I act on it.
    rows = list(results["concepts"])

    def hit_rate(r):
        vt = r["variants_total"] or 0
        return (r["HIT"] / vt) if vt else 0.0

    rows.sort(key=hit_rate)
    print("PER-CONCEPT (sorted by hit rate ascending, weakest first):")
    print("  HIT rate is REACHABILITY (did each seeded phrasing resolve). P is EVICTION")
    print("  RESISTANCE (deposit worth). They are independent by design.")
    header = ("  %-26s %-40s %5s %5s %4s %4s %4s %4s %4s %7s"
              % ("concept", "canonical question", "wrds", "P", "var", "HIT", "MIS", "WRG", "FGN", "rate"))
    print(header)
    for r in rows:
        vt = r["variants_total"] or 0
        rate = ("%.2f" % hit_rate(r)) if vt else "n/a"
        print("  %-26s %-40s %5d %5.2f %4d %4d %4d %4d %4d %7s" % (
            truncate(r["id"], 26),
            truncate(r["question"], 40),
            r["answer_words"],
            r["principal"],
            vt,
            r["HIT"], r["MISS"], r["WRONG"], r["FOREIGN"],
            rate,
        ))
    print("")

    # MISS list: seeds that did not land.
    if results["misses"]:
        print("MISSES (phrasings that did not resolve even after seeding, grouped by concept):")
        by_concept = {}
        for m in results["misses"]:
            by_concept.setdefault(m["concept"], []).append(m)
        for cid in sorted(by_concept):
            print("  concept " + cid + ":")
            for m in by_concept[cid]:
                print("    [MISS] variant=\"" + m["variant"] + "\"  reduced_keywords={"
                      + ", ".join(m["keywords"]) + "}")
                print("           remediation: " + m["note"])
        print("")
    else:
        if mode != "dry-run":
            print("MISSES: none. Every seeded phrasing resolved to its concept.")
            print("")

    # WRONG-concept cross-talk.
    if results["wrong"]:
        print("CROSS-TALK (a variant hit a different seeded concept, disambiguate the wording):")
        for w in results["wrong"]:
            print("  [WRONG] variant=\"" + w["variant"] + "\"  expected=" + w["expected"]
                  + "  got=" + w["got"])
            print("          " + w["expected"] + " and " + w["got"]
                  + " share too many keywords, one needs more distinctive wording.")
        print("")

    # FOREIGN hits.
    if results["foreign"]:
        print("FOREIGN (drill landed on pre-existing production knowledge, not a seed):")
        for f in results["foreign"]:
            print("  [FOREIGN] concept=" + f["concept"] + "  variant=\"" + f["variant"] + "\"")
        print("")

    # Footer.
    t = results["totals"]
    vt = results["variants_total"]
    overall = (t["HIT"] / vt) if vt else 0.0
    print(line)
    print("SUMMARY")
    print("  concepts seeded   : " + str(results["seeded"]))
    if results["seed_failures"]:
        print("  seed failures     : " + ", ".join(results["seed_failures"]))
    print("  variants drilled  : " + str(vt))
    print("  HIT / MISS / WRONG / FOREIGN / ERROR : %d / %d / %d / %d / %d"
          % (t["HIT"], t["MISS"], t["WRONG"], t["FOREIGN"], t["ERROR"]))
    print("  compounding events: " + str(results["compounding_events"]) + " (eviction resistance, not reachability)")
    print("  overall hit rate  : " + (("%.2f" % overall) if vt else "n/a") + " (reachability of seeded phrasings)")
    print("  elapsed           : %.2fs" % elapsed)
    print("  exit code         : " + str(exit_code))
    print(line)
    print("Every deaf child has a right to language. This bank keeps that answer ready.")


def build_json_report(resolved_dir, mode, stats_before, stats_after, results, elapsed, exit_code):
    """Stable-keyed JSON mirror of the report for piping."""
    b, a = stats_before, stats_after
    t = results["totals"]
    vt = results["variants_total"]
    concepts_out = []
    for r in results["concepts"]:
        cvt = r["variants_total"] or 0
        concepts_out.append({
            "id": r["id"],
            "canonical_question": r["question"],
            "answer_words": r["answer_words"],
            "principal_estimate": round(r["principal"], 4),
            "variants_total": cvt,
            "HIT": r["HIT"], "MISS": r["MISS"], "WRONG": r["WRONG"],
            "FOREIGN": r["FOREIGN"], "ERROR": r["ERROR"],
            "hit_rate": round((r["HIT"] / cvt), 4) if cvt else None,
            "seed_ok": r["seed_ok"],
        })
    return {
        "cache_dir": resolved_dir,
        "model": DEFAULT_MODEL,
        "vault": VAULT_NAME,
        "mode": mode,
        "stats_before": stats_before,
        "stats_after": stats_after,
        "delta": {
            "count": a["count"] - b["count"],
            "bytes": a["bytes"] - b["bytes"],
            "hits": a["hits"] - b["hits"],
            "bank": round(a["bank"] - b["bank"], 6),
        },
        "seeded": results["seeded"],
        "seed_failures": results["seed_failures"],
        "concepts": concepts_out,
        "misses": results["misses"],
        "wrong": results["wrong"],
        "foreign": results["foreign"],
        "totals": t,
        "variants_total": vt,
        "compounding_events": results["compounding_events"],
        "compounding_note": "hits and principal drive eviction resistance only, not reachability",
        "reachability_note": "reachability is decided by exact/normalized or content-word Jaccard match, established by seeding",
        "overall_hit_rate": round((t["HIT"] / vt), 4) if vt else None,
        "elapsed_seconds": round(elapsed, 3),
        "exit_code": exit_code,
    }


# -- cache dir resolution -----------------------------------------------------
def resolve_cache_dir(positional, flag_value):
    """First non-empty wins: --cache-dir, then positional, then $BRAIN_CACHE_DIR,
    then the hard default. Returns (resolved_abs_path, source_label, overrode_positional)."""
    overrode = False
    chosen = None
    source = None
    if flag_value:
        chosen = flag_value
        source = "--cache-dir"
        if positional:
            overrode = True
    elif positional:
        chosen = positional
        source = "positional arg"
    elif os.environ.get("BRAIN_CACHE_DIR"):
        chosen = os.environ.get("BRAIN_CACHE_DIR")
        source = "$BRAIN_CACHE_DIR"
    else:
        chosen = DEFAULT_CACHE_DIR
        source = "default"
    resolved = os.path.abspath(os.path.expanduser(chosen))
    return resolved, source, overrode


def confirm_reset(assume_yes):
    """Gate a destructive --reset. Return True only on explicit confirmation."""
    if assume_yes:
        return True
    try:
        if not sys.stdin.isatty():
            return False
    except Exception:
        return False
    try:
        reply = input("This will DELETE every entry in the cache dir. Type y to confirm: ")
    except Exception:
        return False
    return reply.strip().lower() in ("y", "yes")


# -- argument parsing ---------------------------------------------------------
def build_parser():
    epilog = (
        "exit codes:\n"
        "  0  success: primed, dry-run reported, or reset completed with at least one seed.\n"
        "     misses alone are expected output and never make the run non-zero.\n"
        "  2  usage error: bad flags, --reset with --dry-run, or --reset without --yes on a non-tty.\n"
        "  3  import error: answer_cache.py could not be imported from the script's own folder.\n"
        "  4  environment error: the cache dir could not be created or accessed on a live run.\n"
        "  5  data error: every seed store() returned nothing, so no knowledge landed.\n"
        "  1  an unexpected error escaped the guard (should not happen).\n"
    )
    p = argparse.ArgumentParser(
        prog="prime_cache.py",
        description="Warm the Jetson brain's AnswerCache with grounded Deaf-scholarship knowledge.",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("cache_dir_pos", nargs="?", default=None, metavar="cache_dir",
                   help="optional cache dir (positional). --cache-dir overrides it.")
    p.add_argument("--cache-dir", dest="cache_dir_flag", default=None,
                   help="explicit cache dir, overrides positional and $BRAIN_CACHE_DIR.")
    p.add_argument("--reset", action="store_true",
                   help="clear the cache dir before seeding (guarded, needs --yes on a non-tty).")
    p.add_argument("--yes", action="store_true",
                   help="non-interactive confirmation for --reset.")
    p.add_argument("--dry-run", action="store_true",
                   help="report the plan only, no store() and no lookup() (lookup mutates).")
    p.add_argument("--concept", dest="concepts", action="append", default=None, metavar="ID",
                   help="prime only this concept id (repeatable). Unknown ids are warned and skipped.")
    p.add_argument("--quiet", action="store_true",
                   help="suppress per-variant lines, print the final summary only.")
    p.add_argument("--json", dest="as_json", action="store_true",
                   help="emit the final report as a single JSON object on stdout.")
    return p


# -- main ---------------------------------------------------------------------
def main():
    start = time.time()
    parser = build_parser()
    # argparse exits 2 on unknown/extra argv, which matches the usage-error code.
    args = parser.parse_args()

    resolved_dir, source, overrode = resolve_cache_dir(args.cache_dir_pos, args.cache_dir_flag)
    quiet = args.quiet
    as_json = args.as_json
    # In --json mode stdout must be pure JSON for piping, so every human line is
    # suppressed and the per-variant drill is run silently. chatter gates all
    # informational stdout; errors always go to stderr regardless.
    chatter = not quiet and not as_json
    loop_quiet = quiet or as_json

    if overrode and chatter:
        print("Note: --cache-dir wins, overriding the positional cache dir.")

    # --reset and --dry-run contradict: dry-run promises no writes.
    if args.reset and args.dry_run:
        sys.stderr.write("--reset and --dry-run contradict each other. Nothing was written.\n")
        sys.stdout.flush()
        return 2

    if chatter:
        print(BANNER)
        print("")
        print("cache_dir resolved to: " + resolved_dir + " (source: " + source + ")")

    # Filter concepts if --concept was given.
    concepts = KNOWLEDGE
    if args.concepts:
        wanted = list(args.concepts)
        known_ids = {c["id"] for c in KNOWLEDGE}
        for cid in wanted:
            if cid not in known_ids and chatter:
                print("Warning: unknown concept id \"" + cid + "\", skipping it.")
        concepts = [c for c in KNOWLEDGE if c["id"] in set(wanted)]
        if not concepts:
            sys.stderr.write("No known concept ids were selected. Nothing to do.\n")
            sys.stdout.flush()
            return 2

    # Preflight the cache dir. In a live run an unavailable dir is fatal (exit 4)
    # so we never prime a phantom in-memory cache that is discarded on exit. In
    # dry-run it is only a warning, since dry-run writes nothing.
    dir_ok = True
    dir_reason = ""
    try:
        os.makedirs(resolved_dir, exist_ok=True)
    except Exception as e:
        dir_ok = False
        dir_reason = str(e)

    if not dir_ok and not args.dry_run:
        sys.stderr.write("cache_dir unavailable: " + resolved_dir + " (" + dir_reason + ")\n")
        sys.stdout.flush()
        return 4
    if not dir_ok and args.dry_run and chatter:
        print("Warning: cache_dir is not writable (" + dir_reason + "), but dry-run writes nothing.")

    # PHASE 0: construct the cache and record the starting bank.
    try:
        cache = AnswerCache(resolved_dir)
    except Exception as e:
        # AnswerCache is never-raise by contract, but I stay defensive.
        sys.stderr.write("Could not construct AnswerCache: " + repr(e) + "\n")
        sys.stdout.flush()
        return 4

    stats_before = safe_stats(cache)

    mode = "dry-run" if args.dry_run else ("reset" if args.reset else "live")

    # Optional guarded reset (never in dry-run, already excluded above).
    if args.reset:
        if not confirm_reset(args.yes):
            sys.stderr.write("refusing --reset without --yes (non-interactive)\n")
            sys.stdout.flush()
            return 2
        try:
            removed = cache.clear()
        except Exception as e:
            removed = 0
            if chatter:
                print("Reset ran into trouble (" + repr(e) + "), continuing to seed.")
        if chatter:
            print("cleared " + str(int(removed)) + " entries from " + resolved_dir)
        # Refresh the baseline so bank delta reflects the post-clear state.
        stats_before = safe_stats(cache)

    if chatter:
        if args.dry_run:
            print("")
            print("Dry-run plan (no store, no lookup, zero side effects):")
        else:
            print("")
            print("Priming " + str(len(concepts)) + " concept(s). Seeding canonical plus each")
            print("phrasing for reachability, then drilling to compound and verify:")

    # Run the loop. A hard invariant: dry_run reaches neither store() nor lookup().
    # The loop runs silent under --quiet or --json (JSON stdout must stay pure).
    results = prime(cache, concepts, args.dry_run, loop_quiet)

    # Refresh stats after the run.
    stats_after = safe_stats(cache)

    # Decide the exit code.
    if args.dry_run:
        exit_code = 0
    elif results["seeded"] == 0:
        # Every canonical store() returned None: nothing landed.
        exit_code = 5
    else:
        exit_code = 0

    # Report.
    if as_json:
        payload = build_json_report(resolved_dir, mode, stats_before, stats_after,
                                    results, time.time() - start, exit_code)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print_report(resolved_dir, mode, stats_before, stats_after,
                     results, time.time() - start, exit_code)

    sys.stdout.flush()
    return exit_code


if __name__ == "__main__":
    # Outer never-crash guard. No raw traceback ever reaches the operator.
    try:
        code = main()
        try:
            sys.stdout.flush()
        except Exception:
            pass
        sys.exit(code if isinstance(code, int) else 0)
    except SystemExit:
        raise
    except Exception as exc:
        sys.stderr.write("unexpected error: " + repr(exc) + "\n")
        try:
            sys.stdout.flush()
        except Exception:
            pass
        sys.exit(1)


# --- Parent-resource concepts (2026-07-18): grounded in the vault content
# pack (EHDI, IFSP, Clerc Center, NDC) and the re-verified MSD note; authored
# and adversarially verified against the notes. maryland_school_deaf is
# refreshed in place (HB1052 board change, verified Sun commentary).
_PARENT_ENTRIES = [
    {
        "id": "ehdi_early_intervention",
        "canonical_question": "EHDI newborn hearing screening 1-3-6 benchmarks track audiology follow-up, not language outcomes",
        "answer": "EHDI stands for Early Hearing Detection and Intervention. It is the United States system behind the hearing screening every newborn receives, and every state and territory runs a program. About 2 to 3 of every 1,000 babies are born deaf or hard of hearing, and more than 90 percent of them are born to hearing parents, so for most families that screening result is their first experience with deafness. The timeline you will hear is called 1-3-6: screening completed by 1 month, a full audiologic diagnosis by 3 months, and enrollment in early intervention by 6 months. It comes from the Joint Committee on Infant Hearing, and the committee itself says the goal is language competence, in spoken language, signed language, or both.\n\nNow the limitation I want every parent to understand. Every EHDI benchmark measures audiological process: screened, diagnosed, enrolled. None of them measures whether your child is actually acquiring language. A child can hit 1, 3, and 6 right on time and still grow up language deprived if what follows enrollment is not accessible language. Wyatte Hall's research puts it plainly: language deprivation, not hearing loss itself, is the central risk to a deaf child's development, and a cochlear implant is not a reliable standalone first-language intervention.\n\nSo use EHDI and hold it to its deadlines. Then go further. Ask your early intervention program, in writing, how they will track your child's language milestones in both ASL and English. Ask for a Deaf mentor for your family. EHDI counts enrollment. You must count language.",
        "keyword_variants": [
            "EHDI 1-3-6 benchmarks explained",
            "my newborn failed the hearing screening at the hospital, what happens next?",
            "what does Early Hearing Detection and Intervention actually track",
            "newborn hearing screening timeline 1 month 3 months 6 months",
            "does the EHDI system measure language outcomes or just audiology follow-up",
            "baby got a refer result on the hospital hearing test, what is the 1-3-6 timeline",
            "Joint Committee on Infant Hearing position statement benchmarks",
            "can a baby meet every EHDI benchmark on time and still fall behind in language",
            "what should I ask my state EHDI program and early intervention after a failed newborn screen",
            "lost to follow-up after newborn hearing screening diagnosis stage",
            "newborn hearing screen refer what do parents do next",
            "EHDI enrollment by 6 months but nobody is tracking my baby's language milestones",
            "JCIH 1-3-6 versus 1-2-3 timeline",
            "is passing the newborn hearing screening milestones enough for my deaf baby",
            "how the US newborn hearing screening system works state EHDI programs"
        ],
        "sources": [
            "EHDI Early Hearing Detection and Intervention",
            "What Is Language Deprivation",
            "First 90 Days After Diagnosis"
        ]
    },
    {
        "id": "ifsp_part_c_early_intervention",
        "canonical_question": "What is an IFSP, and what should a family of a deaf baby ask for in IDEA Part C birth-to-three early intervention?",
        "answer": "When your baby is identified as deaf, the law is already on your side. Part C of IDEA gives every family a right to early intervention from birth until the third birthday, and the plan that drives it all is called an IFSP, the Individualized Family Service Plan. The clock moves fast on purpose: from referral, the evaluation, the assessments, and the first IFSP meeting must be completed within 45 days.\n\nAn IFSP is not an IEP. An IEP is a school-based education plan that begins at age three. An IFSP is family-centered. It includes outcomes for your family as well as your child, and most services happen in your own home. The written plan must spell out your child's present levels of development, measurable outcomes with criteria and timelines, and each service with its frequency and intensity, and it must be reviewed at least every six months.\n\nYou are a required member of the team, with rights to prior written notice, informed consent, and dispute options if the program says no. So put your requests in the plan, in writing. Ask for ASL instruction for your whole family as family training. Ask for a Deaf mentor: a Deaf adult, fluent in ASL, who visits your home, teaches sign in daily routines, and shares Deaf culture. Ask for early interventionists with real experience with deaf infants, not generic providers. And make language the headline outcome, in ASL, spoken English, or both. Children in bilingual-bicultural programming like this showed considerably larger vocabularies and stronger language.",
        "keyword_variants": [
            "what is an IFSP",
            "IFSP vs IEP what is the difference",
            "IDEA Part C early intervention for a deaf baby",
            "birth to three services for my deaf child",
            "what to ask for at the IFSP meeting",
            "can early intervention pay for ASL classes for my family",
            "getting a Deaf mentor in my baby's early intervention plan",
            "my baby did not pass the hearing screening what services come next",
            "45 day timeline after early intervention referral",
            "our rights if the early intervention program refuses a service",
            "Individualized Family Service Plan explained for parents",
            "what happens when Part C ends at age 3",
            "early intervention providers who actually know deaf infants",
            "how often is the IFSP reviewed"
        ],
        "sources": [
            "IFSP and Part C Early Intervention",
            "Finding Deaf Mentors and Community",
            "EHDI Early Hearing Detection and Intervention"
        ]
    },
    {
        "id": "national_family_orgs",
        "canonical_question": "Which national organizations help families with a deaf child, and what does each one actually offer?",
        "answer": "Yes, and the good news is that the strongest ones are free to any family in the United States, whether or not your child is enrolled anywhere. Here is who they are and what each actually does.\n\nThe National Association of the Deaf, the NAD, is the oldest Deaf civil-rights organization in the United States, run by Deaf people since 1880 and headquartered in Silver Spring, Maryland. For families, its value is authority and advocacy: it publishes position statements on early intervention, cochlear implants, education, and language rights that you can bring to an IFSP or IEP meeting, and its state associations run community events where your child meets Deaf adults.\n\nThe Laurent Clerc National Deaf Education Center, at Gallaudet University in Washington DC, is the national center for deaf education. It offers national family resources at no cost: parent toolkits, early intervention materials, ASL learning resources, and Odyssey magazine. It also runs two tuition-free demonstration schools on the Gallaudet campus, KDES for elementary and MSSD for high school, and you do not need to live in DC or enroll your child to use its family resources.\n\nThe National Deaf Center on Postsecondary Outcomes, federally funded at the University of Texas, focuses on what comes after high school: it publishes the national data on deaf education and employment outcomes and offers free transition-planning tools families use as their child approaches adulthood.\n\nAdd the American Society for Deaf Children for parent-to-parent support and family camps, and Hands & Voices for communication-neutral parent advocacy. All of these serve families anywhere in the country; none requires enrollment or payment for their family resources.",
        "keyword_variants": [
            "national organizations that help families with a deaf child",
            "what national deaf organizations offer parents",
            "are there national groups for parents of deaf children and what does each do",
            "NAD Clerc Center National Deaf Center what do they each offer families",
            "national resources for families of deaf kids free",
            "which deaf organizations should new parents know about",
            "national deaf advocacy and family support organizations list",
            "who helps families of deaf children at the national level",
            "national organizations deaf child parents overview",
            "NAD versus Clerc Center versus National Deaf Center for families",
            "free national family resources deaf education organizations",
            "big deaf organizations every hearing parent should know",
        ],
        "sources": [
            "Where to Find Help National Organizations",
            "National Association of the Deaf",
            "Laurent Clerc National Deaf Education Center",
            "National Deaf Center",
        ]
    },
    {
        "id": "dmv_local_resources",
        "canonical_question": "What local services, schools, and deaf resources exist in the DC and Maryland area for a family with a deaf child?",
        "answer": "Your family happens to live in one of the best regions in the country to raise a deaf child, and the pathway is concrete. Start with early intervention: in Maryland, the Infants and Toddlers Program takes referrals from any parent at 1-800-535-0182 or online at online.mditp.org, and in DC the program is OSSE Strong Start. Part C of IDEA makes these services free and your legal right from birth to age three, with a 45-day clock from referral to a written family service plan; ask in writing for family ASL instruction, a Deaf mentor, and providers experienced with deaf infants.\n\nWhen you do not know which agency you need, use the front doors: the Maryland Governor's Office of the Deaf and Hard of Hearing (odhh.maryland.gov, videophone 443-453-5761) and DC's Division of Deaf, DeafBlind and Hard of Hearing exist to route families to the right service.\n\nFor school, the region holds two of the strongest options anywhere: the Maryland School for the Deaf, an ASL and English bilingual school with campuses in Frederick and Columbia whose Family Education and Partners for Success programs serve families statewide from birth, and the Clerc Center at Gallaudet in Washington DC, whose tuition-free demonstration schools KDES and MSSD serve elementary through high school. County public school DHH programs exist across the region as well.\n\nBeyond school: the National Association of the Deaf is headquartered in Silver Spring, and Maryland and DC both have state Deaf associations, Deaf churches, and social groups where your child meets Deaf adults. Maryland Hands & Voices runs parent-to-parent support and Guide By Your Side. Hearing parents learn ASL at Gallaudet, MSD family classes, and community colleges across the region. Deaf summer camps, from Maryland-based camps to national programs, keep language growing between school years. The vault's Local Resources DMV notes hold the full verified directory with contacts.",
        "keyword_variants": [
            "deaf services near me DC Maryland",
            "local resources for deaf child in Maryland",
            "what programs exist for deaf children in Washington DC",
            "Maryland early intervention deaf baby referral phone",
            "DC services for families with a deaf child",
            "deaf schools and programs in the DMV area",
            "where do I start locally with a deaf baby in Maryland",
            "Montgomery County Prince Georges deaf child services",
            "summer camps for deaf kids near DC",
            "ASL classes for parents in Maryland or DC",
            "Maryland Office of the Deaf and Hard of Hearing help",
            "deaf community events near Washington DC for families",
            "who do I call in Maryland after my baby failed the hearing test",
            "DMV area deaf family resources directory",
        ],
        "sources": [
            "Where to Find Help DC and Maryland Local Services",
            "Early Intervention DC Maryland",
            "Schools and Programs for Deaf Children DC Maryland",
            "State Deaf Offices DC Maryland",
        ]
    },
    {
        "id": "newly_identified_first_steps",
        "canonical_question": "We just found out our baby is deaf and feel overwhelmed; what should we do first?",
        "answer": "Take a breath. What you are feeling is normal, and none of this is your fault: more than 90 percent of deaf babies are born to hearing parents, so nearly every family starts exactly where you are, with no map. Your baby is healthy and whole, and deafness is a difference with its own language and community, not a tragedy. Here is where to start.\n\nFirst, start visible language today. You do not need to be fluent for your signing to count; babies need to SEE language early, and parents who begin signing immediately, however imperfectly, protect their child's development. Every day of accessible language in the first year matters, because the critical period for language acquisition is at its most powerful now and does not wait for anyone.\n\nSecond, contact your state's early intervention program this week. Under Part C of IDEA every family has a legal right to early intervention from birth, at no cost, and the referral starts the clock on evaluations and a written family service plan.\n\nThird, when services begin, ask for bilingual-bimodal support: ASL and English together, not either-or. Ask for a Deaf mentor, a Deaf adult who comes to your home, models fluent signing for your baby, and coaches your whole family. Meeting Deaf adults early is one of the most informative things a new family can do.\n\nDo not let anyone talk you into waiting to see how hearing technology performs before you sign. Hearing aids and implants can be part of the plan, but sign language is the only guaranteed access while everything else is being figured out. You have not lost time. You are starting now, and starting now works.",
        "keyword_variants": [
            "just found out our baby is deaf where do we start",
            "newborn diagnosed deaf overwhelmed parents first steps",
            "my baby failed the hearing test and is deaf, what now",
            "what should hearing parents do first for a deaf baby",
            "new deaf baby diagnosis what to do this week",
            "first steps after learning your child is deaf",
            "baby is deaf and I do not know anything about deafness",
            "we are grieving the diagnosis is our baby going to be okay",
            "should we wait and see before learning sign language",
            "how soon do we need to start signing with our deaf newborn",
            "deaf infant what services and who do we call",
            "overwhelmed new parent of a deaf child needs a starting point",
            "is it our fault our baby was born deaf",
            "what matters most in the first year for a deaf baby",
        ],
        "sources": [
            "First 90 Days After Diagnosis",
            "What Is Language Deprivation",
            "IFSP and Part C Early Intervention",
            "Finding Deaf Mentors and Community",
        ]
    },
    {
        "id": "deaf_mentor_pathways",
        "canonical_question": "Why does my deaf child need to know Deaf adults, and how does a family actually find Deaf mentors?",
        "answer": "Because you cannot be everything, and you do not have to be. Hearing parents who learn ASL are doing something wonderful, but while you are learning, your child also needs fluent language models, and that is what Deaf adults provide. A Deaf mentor fills the language-quality gap during exactly the years it matters most, and gives your child something no hearing person can: a living example of Deaf adulthood, of language, culture, and identity carried in one person. Ask Deaf adults what they wish every hearing parent knew, and you will learn more in an afternoon than in a stack of pamphlets; meeting Deaf adults is one of the most informative things a new family can do.\n\nNow the practical part: how to find them. If your child is under three, use the law first. Families can request Deaf mentor services through early intervention, written into the IFSP, the family service plan every eligible child has a right to under IDEA Part C. Ask for the mentor by name as a service, with visits scheduled in the written plan, and if the program says it has no Deaf mentors on staff, ask them in writing to locate one.\n\nBeyond early intervention: many states run Deaf Mentor Programs through their health department or school for the deaf. The American Society for Deaf Children makes mentor referrals and runs family camps. Your state school for the deaf hosts parent-infant classes, family ASL classes, and community nights, and employs many of the fluent signers in your region. And Deaf community events, coffee nights, storytelling, sports, are open doors: show up, sign imperfectly, keep coming back. Deaf community events and mentor programs are the concrete ways families connect, and the welcome deepens with every visit.",
        "keyword_variants": [
            "why should my deaf child know Deaf adults",
            "how do we find a Deaf mentor for our family",
            "people say my deaf son needs Deaf role models why",
            "Deaf mentor program early intervention IFSP request",
            "where do hearing parents meet Deaf adults",
            "Deaf role model for deaf toddler how to arrange",
            "can early intervention provide a Deaf mentor",
            "my child has never met a deaf adult is that a problem",
            "connecting our family with the Deaf community",
            "Deaf mentor benefits language development deaf child",
            "state Deaf mentor programs for families",
            "how important are Deaf adults in a deaf child's life",
            "finding Deaf community events for families with a deaf baby",
            "ASDC mentor referral deaf children families",
        ],
        "sources": [
            "Finding Deaf Mentors and Community",
            "IFSP and Part C Early Intervention",
            "Deaf Community Cultural Transmission",
        ]
    },
    {
        "id": "sign_speech_myth",
        "canonical_question": "Is there any truth to the claim that signing with a deaf toddler will prevent or delay learning to talk?",
        "answer": "No. This is one of the oldest and most damaging myths in deaf education, and the research does not support it. Study after study finds that early sign language exposure has either no effect on spoken-language development or a positive one: children with a strong first language have a stronger foundation for every language that follows, spoken English included. Signing does not compete with speech; it feeds the same language system.\n\nIt helps to know where the claim actually comes from. It did not come from data. It comes from oralist advocacy traditions, more than a century of campaigning to keep sign language away from deaf children so they would be forced to speak, a movement driven by ideology about normalcy rather than by evidence about outcomes. The claim survives because it gets repeated by people who mean well, including doctors and relatives, not because anyone has demonstrated it.\n\nHere is the frame that serves your son: what matters most is that he acquires a full first language during the critical period, the early window when the brain builds language most easily. A complete first language in those years benefits him for life regardless of which language he later prefers, and gambling those years on speech alone is the one choice the evidence warns against. Your mother-in-law loves her grandson and is repeating what she has heard, so share the evidence gently: signing with him now is not holding him back. It is the safest thing you can do for his speech, his reading, and his mind.",
        "keyword_variants": [
            "will signing stop my deaf toddler from learning to talk",
            "grandmother says sign language prevents speech is that true",
            "does ASL delay spoken language development",
            "myth that signing keeps deaf kids from speaking",
            "relative warns us not to sign so our child will talk",
            "is there evidence sign language harms speech",
            "sign language versus speech development deaf toddler",
            "does teaching a deaf baby to sign hurt talking later",
            "oralism claim sign delays speech research",
            "if we sign will he never talk",
            "speech therapist says signing is a crutch",
            "early sign exposure effect on spoken language outcomes",
            "family pressure to stop signing with deaf child",
            "does bilingual sign and speech confuse a deaf toddler",
        ],
        "sources": [
            "What Is Language Deprivation",
            "Bilingual Bimodal Protective Factors",
            "Grosjean 2001 - The Right of the Deaf Child to Grow Up Bilingual",
        ]
    },
    {
        "id": "deaf_school_vs_mainstream",
        "canonical_question": "What does a school for the deaf offer a deaf child that a mainstream classroom with an interpreter does not?",
        "answer": "The difference is direct access versus filtered access. In a mainstream classroom, everything your daughter gets passes through one interpreter: one voice at a time, a beat behind, during instruction. At a school for the deaf she has direct access to teachers and classmates in a shared visual language, no waiting, no middleman; she can interrupt, joke, argue, and ask the follow-up question herself.\n\nJust as important is everything that happens outside the lesson. Most of what children learn at school is incidental and social: the hallway conversation, the lunchroom debate, the overheard explanation. An interpreter covers the lesson; it cannot cover the day. At a deaf school the incidental and social learning is accessible all day long, which is where vocabulary, social judgment, and identity actually grow.\n\nFor a child who is behind in language, this matters even more: an immersive signing environment is the central remediation the field can offer, and a deaf school is that environment at full strength, with Deaf adults as everyday models of what your daughter can become. These schools are real and close: the Maryland School for the Deaf runs bilingual campuses in Frederick and Columbia, and the Clerc Center at Gallaudet in Washington DC runs tuition-free demonstration schools, KDES for elementary and MSSD for high school.\n\nOne misconception to set down: deaf schools do not reject English, speech, or reading. They pair ASL with English literacy deliberately, and a child who wants spoken-language support gets it there too. Choosing immersion is not choosing against English. It is choosing full access to both.",
        "keyword_variants": [
            "school for the deaf versus mainstream classroom with interpreter",
            "what does a deaf school offer that mainstreaming does not",
            "should our deaf daughter go to a deaf school or public school",
            "is an interpreter in a regular classroom enough for a deaf child",
            "deaf school benefits direct communication peers",
            "incidental learning deaf child mainstream classroom",
            "deaf institute or mainstream placement decision",
            "will a deaf school hurt my child's English",
            "do schools for the deaf teach reading and speech",
            "deaf school immersion language delayed child",
            "Maryland School for the Deaf versus local public school",
            "mainstream with interpreter social isolation deaf student",
            "choosing school placement for deaf kindergartner",
            "deaf school ASL English bilingual education what to expect",
        ],
        "sources": [
            "Maryland School for the Deaf",
            "Laurent Clerc National Deaf Education Center",
            "Starting Late Ages 5 and Up",
        ]
    },
    {
        "id": "late_start_remediation",
        "canonical_question": "Is it too late for an older deaf child who started sign language late, and what remediates a late start?",
        "answer": "No, it is not too late, and the fact that you are asking means your family is already moving. The research on late signers holds two truths at once: earlier is better, and later still helps. Deaf children and teenagers who gain real access to sign language late in childhood still show meaningful gains in language, in behavior, in mental health, and in wellbeing. The window narrows with age; it does not slam shut.\n\nBe gentle with yourselves about how you got here. Language deprivation is almost never the product of neglect; it is usually the unintended outcome of well-intentioned choices, made on professional advice, by parents doing their best with what they were told. The framing that helps your child is forward-looking: not whose fault the gap is, but how fast it can close.\n\nOn that, the evidence points one direction: immersion. The strongest remediation for a late start is an immersive signing environment, all day, every day, among fluent signers, which is exactly what a school for the deaf provides. A delayed child in a mainstream room with one interpreter gets a filtered slice of language; the same child at a school like the Maryland School for the Deaf swims in it, in class, at lunch, on the playground, with ASL paired to English literacy the whole way. Delayed language milestones in a deaf child are an emergency deserving immediate action, and because the efficient years are already spent, every month of full access matters more now, not less. Escalate access without delay: visit your state school for the deaf, call an IEP meeting about direct ASL instruction, bring Deaf adults into your child's life, and put the whole family in ASL classes. Starting late is not the end of the story. Stopping is.",
        "keyword_variants": [
            "is it too late for my deaf child to learn sign language",
            "we started signing late and I feel terrible about waiting",
            "my deaf son is seven and we only started signing this year",
            "older deaf child with almost no language what do we do",
            "late first language exposure outcomes for deaf kids",
            "can a deaf teenager still acquire ASL",
            "language deprivation remediation for an older child",
            "started ASL late is there still hope",
            "deaf child far behind in language milestones emergency",
            "school for the deaf for a language delayed late start child",
            "my child is ten and barely has any language",
            "late signers still gain language behavior wellbeing",
            "how to catch up a deaf child who started sign language late",
            "we waited too long to sign with our deaf child is he ruined",
            "does starting sign at age eight still work",
        ],
        "sources": [
            "Starting Late Ages 5 and Up",
            "What Is Language Deprivation",
            "Maryland School for the Deaf",
            "Glickman and Hall 2018 - Language Deprivation and Deaf Mental Health",
        ]
    },
    {
        "id": "clerc_center",
        "canonical_question": "What is the Laurent Clerc National Deaf Education Center, with its tuition-free KDES and MSSD demonstration schools and free national resources for families of deaf children?",
        "answer": "The Laurent Clerc National Deaf Education Center, usually just called the Clerc Center, is the national center for deaf education, based at Gallaudet University in Washington, DC. It is federally funded under the Education of the Deaf Act, and its charge is national: developing and sharing what works in deaf education, and supporting the parents and professionals who serve deaf and hard of hearing children from birth to age 21. Its early intervention mission is plain: end language deprivation by making sure deaf babies have full access to both ASL and English from the start.\n\nThe center runs two tuition-free demonstration schools on the Gallaudet campus. Kendall Demonstration Elementary School, KDES, is an accredited day school serving children from birth through grade 8 in the DC metro area, beginning with a Parent-Infant Program. The Model Secondary School for the Deaf, MSSD, is an accredited residential high school for grades 9 through 12 that draws students from all 50 states and US territories.\n\nHere is the part many families miss: you do not need to live in DC, and your child does not need to be enrolled, to use the Clerc Center. Its national resources are free. A parent anywhere in the US can join Deaf Education Connections for family videos, toolkits, and research, use early intervention offerings like ASL classes, Deaf mentorship, and bilingual storybooks, get Odyssey magazine by mail at no cost, browse the summer camp listings, and point their local school district to the State and School Partnership program.",
        "keyword_variants": [
            "what is the Clerc Center",
            "Laurent Clerc National Deaf Education Center",
            "Kendall Demonstration Elementary School KDES",
            "Model Secondary School for the Deaf MSSD",
            "tuition free residential high school for deaf teens",
            "federally funded national center for deaf education",
            "free national resources for parents of a deaf child",
            "Odyssey magazine free deaf education",
            "Deaf Education Connections videos and toolkits for families",
            "deaf high school my child can attend from anywhere in the country",
            "Clerc Center early intervention resources for families",
            "free sign language classes and Deaf mentors for parents just starting out",
            "national program that coaches school districts on deaf education",
            "summer camps for deaf and hard of hearing children"
        ],
        "sources": [
            "Laurent Clerc National Deaf Education Center",
            "Where to Find Help National Organizations"
        ]
    },
    {
        "id": "national_deaf_center",
        "canonical_question": "What is the National Deaf Center, and what free national outcome data and transition planning resources does it offer deaf youth and their families?",
        "answer": "The National Deaf Center on Postsecondary Outcomes, usually just called the National Deaf Center or NDC, is a federally funded center at the University of Texas at Austin, supported by the US Department of Education. Think of it as the national clearinghouse for what actually happens to deaf students after high school. Its mission is simple: Learn. Connect. Transform. Everything it offers is deaf-centered, evidence-based, and free, and it serves every US state and territory.\n\nNDC's data give families the long view. In 2024, 1.9 percent of people in the United States are deaf. Just over half of deaf people, 55.8 percent, have some college credit or a degree. Deaf youth ages 16 to 24 are disconnected from both school and work at a higher rate, 22.7 percent, than the national average of 12.6 percent. And deaf people who earn a college degree earn more, work more, and are more involved in their communities. An interactive dashboard lets you see figures for your own state.\n\nFor families, NDC offers free transition planning resources built around a student-led process: deaf youth set their own goals and learn their rights and how to request accommodations, often through the IEP or vocational rehabilitation. The Promoting #DeafSuccess family guide shows that family support and high expectations measurably raise the odds of success, and Deafverse, a free online game, lets deaf teens build self-advocacy skills. A Help Team answers individual questions at no cost.",
        "keyword_variants": [
            "what is the National Deaf Center",
            "National Deaf Center on Postsecondary Outcomes",
            "NDC University of Texas Austin deaf center",
            "national data on deaf education and employment outcomes",
            "statistics on deaf people going to college and getting jobs",
            "what percent of deaf people have a college degree",
            "deaf youth disconnected from school and work rates",
            "transition planning for my deaf teenager after high school",
            "free resources to help a deaf teen get ready for college and work",
            "IEP transition meeting resources for deaf students",
            "Deafverse game for deaf teens",
            "Promoting DeafSuccess guide for families",
            "help my deaf high schooler plan for life after graduation",
            "free federal center for deaf student outcomes after high school"
        ],
        "sources": [
            "National Deaf Center",
            "Where to Find Help National Organizations"
        ]
    }
]
KNOWLEDGE.extend(_PARENT_ENTRIES)

