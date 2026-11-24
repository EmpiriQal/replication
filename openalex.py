"""Requests to OpenAlex (https://api.openalex.org), the open index of scholarly works.

The API key is read from the environment variable OPENALEX_API_KEY and the contact address from OPENALEX_MAILTO. Without a key, every request from the machine's IP address shares one small free daily budget, which a search of a few hundred papers exhausts; with a key, the budget is the key's own.
"""

import os
import time

import httpx

BASE = "https://api.openalex.org"
RETRY_STATUS = (429, 500, 502, 503, 504)


class OpenAlexError(Exception):
    pass


def short_id(url: str) -> str:
    """"W2735610849" from "https://openalex.org/W2735610849"."""
    return url.rsplit("/", 1)[1]


def get(path: str, params: dict | None = None, client=None, tries: int = 9) -> dict:
    """One GET request (path relative to the API root), retried with growing waits on rate limits and server errors."""
    params = dict(params or {})
    for env, name in (("OPENALEX_API_KEY", "api_key"), ("OPENALEX_MAILTO", "mailto")):
        if os.environ.get(env):
            params[name] = os.environ[env]
    send = client.get if client is not None else httpx.get
    for attempt in range(tries):
        r = send(f"{BASE}/{path}", params=params, timeout=60)
        if r.status_code == 200:
            return r.json()
        if r.status_code not in RETRY_STATUS:
            raise OpenAlexError(f"OpenAlex {path} returned {r.status_code}: {r.text[:200]}")
        time.sleep(min(2 ** (attempt + 1), 120))
    raise OpenAlexError(f"OpenAlex {path} failed after {tries} tries")


def abstract_text(inverted: dict | None) -> str:
    """The abstract from OpenAlex's inverted index (each word mapped to its positions); empty if there is none."""
    if not inverted:
        return ""
    return " ".join(w for _, w in sorted((p, w) for w, ps in inverted.items() for p in ps))


def work_by_doi(doi: str, client=None) -> dict:
    """The work with this DOI: id, title, year and the ids of its related works."""
    w = get(f"works/doi:{doi}", {"select": "id,display_name,publication_year,related_works"}, client)
    return {"id": short_id(w["id"]), "title": w["display_name"], "year": w["publication_year"], "related_ids": [short_id(u) for u in w.get("related_works") or []]}


def works(ids: list[str], client=None) -> list[dict]:
    """The works with these OpenAlex ids, in the given order (missing ones left out): id, title, year, publication date and abstract. At most 50 ids per request."""
    out = {}
    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        got = get("works", {"filter": "openalex:" + "|".join(chunk), "per-page": 50, "select": "id,display_name,publication_year,publication_date,abstract_inverted_index"}, client)
        for g in got["results"]:
            out[short_id(g["id"])] = {"id": short_id(g["id"]), "title": g["display_name"], "year": g["publication_year"], "date": g["publication_date"], "abstract": abstract_text(g.get("abstract_inverted_index"))}
    return [out[i] for i in ids if i in out]


def related_works(doi: str, client=None) -> dict:
    """The work with this DOI and its related works (OpenAlex's "related_works", about 10 works chosen by shared topics), each with title, year, date and abstract."""
    w = work_by_doi(doi, client)
    return {"id": w["id"], "title": w["title"], "related": works(w["related_ids"], client) if w["related_ids"] else []}


def citing_works(work_id: str, search: str, before: str | None = None, client=None, limit: int = 2000) -> list[dict]:
    """Works that cite this work and match the search in their title or abstract (OpenAlex syntax, e.g. "replication OR reproduction"), published before the date if one is given (YYYY-MM-DD): id, title, year, date, abstract. Pages through the results, up to `limit` works."""
    flt = f"cites:{work_id},title_and_abstract.search:{search}"
    if before:
        flt += f",to_publication_date:{before}"
    out, cursor = [], "*"
    while cursor and len(out) < limit:
        got = get("works", {"filter": flt, "per-page": 200, "cursor": cursor, "select": "id,display_name,publication_year,publication_date,abstract_inverted_index"}, client)
        for g in got["results"]:
            out.append({"id": short_id(g["id"]), "title": g["display_name"], "year": g["publication_year"], "date": g["publication_date"], "abstract": abstract_text(g.get("abstract_inverted_index"))})
        cursor = got["meta"].get("next_cursor") if got["results"] else None
    return out
