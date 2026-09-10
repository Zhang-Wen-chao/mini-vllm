import sqlite3, sys, json

def analyze(path):
    con = sqlite3.connect(path)
    cur = con.cursor()
    rows = cur.execute("""
        SELECT start, "end" FROM NVTX_EVENTS
        WHERE text = 'steady_decode' ORDER BY start DESC LIMIT 1
    """).fetchall()
    if not rows:
        return {"path": path, "error": "no steady_decode range"}
    ws, we = rows[0]
    kcount, ksum = cur.execute("""
        SELECT COUNT(*), SUM("end" - start) FROM CUPTI_ACTIVITY_KIND_KERNEL
        WHERE start >= ? AND "end" <= ?
    """, (ws, we)).fetchone()
    api = cur.execute("""
        SELECT s.value, COUNT(*), SUM(r."end" - r.start) FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        JOIN StringIds s ON r.nameId = s.id
        WHERE r.start >= ? AND r."end" <= ? GROUP BY s.value ORDER BY 2 DESC
    """, (ws, we)).fetchall()
    top = [{"api": a, "calls": c, "us": round((t or 0)/1e3)} for a, c, t in api[:6]]
    out = {
        "file": path.split("/")[-1],
        "window_ms": round((we - ws)/1e6, 1),
        "kernels": kcount,
        "kernel_ms": round((ksum or 0)/1e6, 2),
        "kernel_share": round(100.0*(ksum or 0)/(we - ws), 1),
        "top_api": top,
    }
    con.close()
    return out

for p in sys.argv[1:]:
    print(json.dumps(analyze(p), ensure_ascii=False))
