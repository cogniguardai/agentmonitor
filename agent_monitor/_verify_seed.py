"""Tiny verification helper — print the seeded runs with no unicode."""
from agent_monitor import db, DB_PATH

with db.session() as conn:
    rows = conn.execute("""
        SELECT r.id, r.external_id, r.status, r.elapsed_ms,
            (SELECT MAX(score) FROM interp_score WHERE run_id=r.id AND probe='harm')    AS harm_max,
            (SELECT MAX(score) FROM interp_score WHERE run_id=r.id AND probe='refusal') AS ref_max,
            (SELECT COUNT(*)   FROM trace_event   WHERE run_id=r.id)                     AS n_trace,
            r.output_text
        FROM run r ORDER BY r.id
    """).fetchall()
    n_traces = conn.execute("SELECT COUNT(*) FROM trace_event").fetchone()[0]
    n_scores = conn.execute("SELECT COUNT(*) FROM interp_score").fetchone()[0]
    n_mem = conn.execute("SELECT COUNT(*) FROM memory_chunk").fetchone()[0]

print(f"DB: {DB_PATH}")
print()
print(f"  {'#':>3}  {'ext':7s} {'status':7s} {'harm':>5s} {'refus':>5s} {'tr':>3s} {'elaps':>7s}  output")
print(f"  " + "-" * 78)
for r in rows:
    harm = "----" if r["harm_max"] is None else f"{r['harm_max']:.2f}"
    ref  = "----" if r["ref_max"]  is None else f"{r['ref_max']:.2f}"
    flag = " *" if r["harm_max"] and r["harm_max"] >= 0.7 else "  "
    out  = (r["output_text"] or "").replace("\n", " ")[:38]
    elap = (r["elapsed_ms"] or 0) / 1000.0
    print(f"  {r['id']:>3}  {(r['external_id'] or ''):7s} {r['status']:7s} "
          f"{harm:>5s}{flag} {ref:>5s} {r['n_trace']:>3d} {elap:>6.1f}s  {out}")
print()
print(f"  totals: {len(rows)} runs / {n_traces} trace events / "
      f"{n_scores} interp scores / {n_mem} memory chunks")
