"""Verify that every expected_section in test_queries.json exists in chunks.json."""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
chunks = json.load(open(PROJECT_ROOT / "data" / "processed" / "chunks.json", encoding="utf-8"))
queries = json.load(open(PROJECT_ROOT / "data" / "eval" / "test_queries.json", encoding="utf-8"))

chunk_sections = {c["section_number"] for c in chunks if c.get("section_number")}
chunk_ids = {c["chunk_id"] for c in chunks}

print(f"Total chunks: {len(chunks)}")
print(f"Unique sections: {len(chunk_sections)}")
print(f"Total test queries: {len(queries)}")
print()

in_scope = [q for q in queries if not q.get("out_of_scope")]
out_of_scope = [q for q in queries if q.get("out_of_scope")]
print(f"In-scope: {len(in_scope)},  Out-of-scope: {len(out_of_scope)}")
print()

ok = 0
fail = 0
for q in in_scope:
    sec = q["expected_section"]
    cid = q["expected_chunk_id"]
    sec_ok = sec in chunk_sections
    cid_ok = cid in chunk_ids
    status = "OK" if (sec_ok and cid_ok) else "FAIL"
    if status == "FAIL":
        fail += 1
        print(f"  FAIL: section={sec} found={sec_ok}, chunk_id={cid} found={cid_ok}")
    else:
        ok += 1

print(f"\nResult: {ok} OK, {fail} FAIL out of {len(in_scope)} in-scope queries")
