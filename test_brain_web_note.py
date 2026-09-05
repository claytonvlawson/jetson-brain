#!/usr/bin/env python3
"""
test_brain_web_note.py - brain_web's GET /note helpers (brain_web_note.py).

Runs with no server, no index, no Ollama: it builds a throwaway vault in a
temp dir and exercises note_path (the traversal guard) and load_note (the
frontmatter split, title/tags extraction, and the body cap).

    python3 test_brain_web_note.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brain_web_note as n  # noqa: E402

FAILS = []


def check(name, cond):
    print(("  ok   " if cond else "  FAIL ") + name)
    if not cond:
        FAILS.append(name)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        vault = os.path.join(tmp, "vault")
        os.makedirs(os.path.join(vault, "People"))
        os.makedirs(os.path.join(vault, "Sources", "Language Deprivation"))
        outside = os.path.join(tmp, "secret.md")
        with open(outside, "w") as fh:
            fh.write("# not for you\n")

        person = os.path.join(vault, "People", "Tom Humphries.md")
        with open(person, "w") as fh:
            fh.write("---\nname: Tom Humphries\ntags:\n  - Person\n  - deaf-studies\n---\n"
                     "# Tom Humphries\n\nCo-author of [[Deaf in America]].\n")
        paper = os.path.join(vault, "Sources", "Language Deprivation",
                             "Humphries et al 2019 - Teaching Your Child.md")
        with open(paper, "w") as fh:
            fh.write("---\ntitle: Teaching Your Child to Sign\nyear: 2019\ntags: [source, parent-guidance]\n---\n"
                     "Body of the paper note.\n")
        plain = os.path.join(vault, "Plain.md")
        with open(plain, "w") as fh:
            fh.write("No frontmatter here.\n")
        big = os.path.join(vault, "Big.md")
        with open(big, "w") as fh:
            fh.write("x" * (n.MAX_BODY + 500))
        # A symlink inside the vault that points outside it.
        link = os.path.join(vault, "People", "Escape.md")
        os.symlink(outside, link)
        # And a non-markdown file.
        with open(os.path.join(vault, "notes.txt"), "w") as fh:
            fh.write("txt\n")

        print("note_path")
        check("accepts a nested .md", n.note_path(vault, "People/Tom Humphries.md") == os.path.realpath(person))
        check("accepts a URL-encoded path",
              n.note_path(vault, "People/Tom%20Humphries.md") == os.path.realpath(person))
        check("accepts spaces and dashes in folders",
              n.note_path(vault, "Sources/Language Deprivation/Humphries et al 2019 - Teaching Your Child.md")
              == os.path.realpath(paper))
        check("rejects ../ traversal", n.note_path(vault, "../secret.md") is None)
        check("rejects nested ../ traversal", n.note_path(vault, "People/../../secret.md") is None)
        check("rejects encoded ../ traversal", n.note_path(vault, "People/%2e%2e/%2e%2e/secret.md") is None)
        check("rejects an absolute path", n.note_path(vault, outside) is None)
        check("rejects a tilde path", n.note_path(vault, "~/x.md") is None)
        check("rejects backslashes", n.note_path(vault, "People\\..\\secret.md") is None)
        check("rejects a non-.md file", n.note_path(vault, "notes.txt") is None)
        check("rejects a missing file", n.note_path(vault, "People/Nobody.md") is None)
        check("rejects a directory", n.note_path(vault, "People.md") is None)
        check("rejects a symlink that escapes the vault", n.note_path(vault, "People/Escape.md") is None)
        check("rejects empty", n.note_path(vault, "") is None and n.note_path("", "Plain.md") is None)
        check("rejects an empty segment", n.note_path(vault, "People//Tom Humphries.md") is None)

        print("load_note")
        d = n.load_note(vault, "People/Tom%20Humphries.md")
        check("person: title from name:", d and d["title"] == "Tom Humphries")
        check("person: tags lowercased", d and d["tags"] == ["person", "deaf-studies"])
        check("person: frontmatter stripped from body", d and d["body"].startswith("# Tom Humphries"))
        check("person: path is the decoded relative path", d and d["path"] == "People/Tom Humphries.md")
        check("person: not truncated", d and d["truncated"] is False)
        d = n.load_note(vault, "Sources/Language Deprivation/Humphries et al 2019 - Teaching Your Child.md")
        check("paper: title from title:", d and d["title"] == "Teaching Your Child to Sign")
        check("paper: inline list tags", d and d["tags"] == ["source", "parent-guidance"])
        d = n.load_note(vault, "Plain.md")
        check("plain: title falls back to filename", d and d["title"] == "Plain")
        check("plain: body passes through", d and d["body"] == "No frontmatter here.")
        check("plain: no tags", d and d["tags"] == [])
        d = n.load_note(vault, "Big.md")
        check("big: body capped", d and len(d["body"]) == n.MAX_BODY and d["truncated"] is True)
        check("rejected path returns None", n.load_note(vault, "../secret.md") is None)

    print()
    if FAILS:
        print("FAILED: %d check(s): %s" % (len(FAILS), ", ".join(FAILS)))
        sys.exit(1)
    print("All test_brain_web_note checks passed.")


if __name__ == "__main__":
    main()
