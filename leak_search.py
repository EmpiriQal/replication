"""Search for SCORE outcomes made public before 8 January 2026: for each of the 164 papers of the 274 claims, OpenAlex works that report a replication of it or come from SCORE.

Two sources per paper: (1) works citing it, published before 8 January 2026, whose title or abstract mentions replication, reproduction, reanalysis or SCORE; (2) its OpenAlex related works (up to 10). A work is a candidate if its title or abstract marks it as a SCORE document (SCORE_DOC), if it names the SCORE program, or if its title mentions replication, reproduction, reanalysis or a comment and its title or abstract names the paper's first author. Candidates are printed for reading; whether one reports SCORE's outcome was decided by reading them, and the result is data/leaked_claims.csv.

OpenAlex records are cached in results/leak_search.json. An OpenAlex API key, if any, is read from OPENALEX_API_KEY and a contact address from OPENALEX_MAILTO.

Usage: uv run python leak_search.py [--report]
"""

import argparse
import csv
import json
import re

import score as S
import openalex

CACHE = S.ROOT / "results/leak_search.json"
SEARCH = "replication OR replicate OR replicated OR reproduction OR reproducibility OR reanalysis OR SCORE"
BEFORE = "2026-01-07"
TITLE_WORDS = re.compile(r"replicat|reproduc|re-?analys|comment", re.I)
# SCORE's OSF documents: titles of the form "<Author>_<Journal>_<Year>_<code> - ", and other SCORE wording
SCORE_DOC = re.compile(r"^[^ ]+_[^ ]+_\d{4}_\w+ - |Replication - |Data Replication|Computational Reproduction|\bSCORE\b|selected for replication")
SCORE_PROGRAM = re.compile(r"Systematizing Confidence|\bSCORE\b (?:program|project|programme)|DARPA", re.I)


def papers() -> dict:
    """DOI to the paper's rows, citation and first author's surname."""
    out = {}
    with open(S.ROOT / S.CONFIG["data_file"], newline="", encoding="utf-8") as f:
        for i, c in enumerate(csv.DictReader(f), 1):
            p = out.setdefault(c["DOI"], {"rows": [], "paper": c["Paper"], "surname": c["Paper"].split()[0].split(",")[0]})
            p["rows"].append(i)
    return out


def candidate(w: dict, surname: str, paper_id: str) -> str | None:
    if w["id"] == paper_id:
        return None
    t, a = w["title"] or "", w["abstract"] or ""
    if SCORE_DOC.search(t) or SCORE_DOC.search(a[:400]):
        return "SCORE document"
    if SCORE_PROGRAM.search(a) or SCORE_PROGRAM.search(t):
        return "names SCORE"
    if TITLE_WORDS.search(t) and surname.lower() in (t + " " + a).lower():
        return "replication or comment naming the author"
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--report", action="store_true", help="from the cache only")
    a = ap.parse_args()
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    ps = papers()
    for n, (doi, p) in enumerate(ps.items(), 1):
        if doi in cache or a.report:
            continue
        r = openalex.related_works(doi)
        cache[doi] = {"id": r["id"], "title": r["title"], "related": r["related"], "citing": openalex.citing_works(r["id"], SEARCH, BEFORE)}
        if n % 10 == 0:
            CACHE.write_text(json.dumps(cache))
            print(f"  {n} of {len(ps)} papers", flush=True)
    CACHE.write_text(json.dumps(cache))

    total = {"citing": 0, "related": 0}
    found = []
    for doi, p in ps.items():
        c = cache.get(doi)
        if not c:
            continue
        seen = set()
        for src in ("citing", "related"):
            total[src] += len(c[src])
            for w in c[src]:
                why = candidate(w, p["surname"], c["id"])
                if why and w["id"] not in seen and (w["date"] or "9999") < "2026-01-08":
                    seen.add(w["id"])
                    found.append((p, src, why, w))
    print(f"Papers searched: {sum(d in cache for d in ps)} of {len(ps)}; citing works matching the search before 8 January 2026: {total['citing']}; related works: {total['related']}; candidates: {len(found)}")
    for p, src, why, w in sorted(found, key=lambda t: t[0]["rows"][0]):
        print(f"\nrows {', '.join(map(str, p['rows']))} | {p['paper']} | {why} | from {src} | {w['date']} | {w['id']}\n  {w['title']}\n  {w['abstract'][:500]}")


if __name__ == "__main__":
    main()
