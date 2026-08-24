Aug 18. Spent the day fixing something I built, then finding out I'd fixed it wrong.

Background: I run a small RAG brain on a Jetson Orin Nano. It answers questions against my
thesis vault on language deprivation, and when it doesn't know something, it searches the
web and keeps what it finds.

Today I actually looked at what it had been keeping.

Nine pages, written into my vault and indexed, nobody asked. Most were fine. One was a
hearing aid retailer whose entire indexed body was a coupon offer followed by a claim about
hearing loss rates by race. That page was competing for context slots against Humphries and
Hall. My own machine had been diluting my sources with SEO copy for weeks and I hadn't
noticed, because everything it did was technically working.

So, a review queue. Fetches land outside the vault now and wait for me. The answer to your
current question still uses the page live. Only the permanence waits for approval.

Then the part I didn't plan for.

I had the implementation torn apart before trusting it, and it came back with nine blockers
in my own work. My favorite: "reject" did nothing at all. It looked for the page's metadata
in the wrong folder, found nothing, denied nothing, then printed "Rejections are durable"
and exited zero. Every happy path test passed.

Close second. I'd added a guard so rebuilding the index couldn't silently destroy the vector
block. The guard checked whether you passed the flag, not whether any vectors actually came
out. So with the flag set and the embedder down, it wiped them anyway and reported success.
That flag is the exact command my deploy script runs. My fix was worse than the bug it
replaced.

Then the genuinely dumb one. Three agent sessions were working in the same git checkout. One
committed while another had files staged and produced a single commit of eighteen files
under a message describing one of them. Two pieces of that are still stranded in the wrong
commit, and I left them there with a note rather than rewrite history under a live session.
The fix was boring: separate worktrees, so each session gets its own index. Plus a lock on
the deploy script, which had none, and which can otherwise delete files mid transfer and
restart services before the copy finishes.

All of it shipped. The coupon page is gone and denied so it can't come back. Eight notes
left, and now they need my approval instead of my attention.

What sticks with me: every single thing that was broken today printed success.
