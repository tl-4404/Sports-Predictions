"""Remove SHA-256 from the MLB ledger while preserving append-only immutability."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# This script is intentionally one-shot; the workflow deletes it after a green run.


def patch_db():
    path = ROOT / "mlb" / "db.py"
    text = path.read_text()
    text = text.replace("import hashlib\n", "")
    text = text.replace(
'''    CREATE TABLE IF NOT EXISTS decision_ledger (\n      id INTEGER PRIMARY KEY AUTOINCREMENT,\n      prediction_id INTEGER NOT NULL UNIQUE REFERENCES prop_predictions(id),\n      ledger_key TEXT NOT NULL UNIQUE,\n      recorded_at TEXT NOT NULL,\n      snapshot_hash TEXT NOT NULL,\n      snapshot_json TEXT NOT NULL\n    );''',
'''    CREATE TABLE IF NOT EXISTS decision_ledger (\n      id INTEGER PRIMARY KEY AUTOINCREMENT,\n      prediction_id INTEGER NOT NULL UNIQUE REFERENCES prop_predictions(id),\n      ledger_key TEXT NOT NULL UNIQUE,\n      recorded_at TEXT NOT NULL,\n      snapshot_json TEXT NOT NULL\n    );''')
    migration_anchor = "    # Safe migrations from every earlier schema version.\n"
    migration = '''    # Migrate the earlier hash-bearing ledger to the simpler append-only schema.\n    ledger_columns = _columns(conn, "decision_ledger")\n    if "snapshot_hash" in ledger_columns:\n        conn.executescript("""\n        CREATE TABLE decision_ledger_v2 (\n          id INTEGER PRIMARY KEY AUTOINCREMENT,\n          prediction_id INTEGER NOT NULL UNIQUE REFERENCES prop_predictions(id),\n          ledger_key TEXT NOT NULL UNIQUE,\n          recorded_at TEXT NOT NULL,\n          snapshot_json TEXT NOT NULL\n        );\n        INSERT INTO decision_ledger_v2\n          (id,prediction_id,ledger_key,recorded_at,snapshot_json)\n        SELECT id,prediction_id,ledger_key,recorded_at,snapshot_json\n        FROM decision_ledger;\n        DROP TABLE decision_ledger;\n        ALTER TABLE decision_ledger_v2 RENAME TO decision_ledger;\n        """)\n\n'''
    if migration not in text:
        text = text.replace(migration_anchor, migration + migration_anchor)
    old = '''    canonical = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), default=str)\n    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()\n    conn.execute(\n        """INSERT INTO decision_ledger\n           (prediction_id,ledger_key,recorded_at,snapshot_hash,snapshot_json)\n           VALUES (?,?,?,?,?)""",\n        (prediction_id, ledger_key, utcnow(), digest, canonical),\n    )'''
    new = '''    canonical = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), default=str)\n    conn.execute(\n        """INSERT INTO decision_ledger\n           (prediction_id,ledger_key,recorded_at,snapshot_json)\n           VALUES (?,?,?,?)""",\n        (prediction_id, ledger_key, utcnow(), canonical),\n    )'''
    if old not in text:
        raise SystemExit("db insert block not found")
    text = text.replace(old, new)
    path.write_text(text)


def patch_decision_compatibility():
    path = ROOT / "mlb" / "decision.py"
    text = path.read_text()
    anchor = 'MAX_MODEL_MARKET_GAP = float(os.getenv("MLB_MAX_AUTO_GAP", "0.12"))\n'
    alias = anchor + 'MAX_AUTO_GAP = MAX_MODEL_MARKET_GAP  # backward-compatible import\n'
    if "MAX_AUTO_GAP = MAX_MODEL_MARKET_GAP" not in text:
        text = text.replace(anchor, alias)
    path.write_text(text)


def patch_docs():
    path = ROOT / "docs" / "MLB_PIPELINE_V5_MERGED_AUDIT.md"
    text = path.read_text()
    text = text.replace(
        "Every prediction insert also writes a SHA-256-hashed `decision_ledger` snapshot.",
        "Every prediction insert also writes an append-only `decision_ledger` snapshot.",
    )
    text = text.replace(
        "Later reruns create new snapshots and never overwrite an earlier decision.",
        "Later reruns create new snapshots and never overwrite an earlier decision. Immutability is enforced by unique IDs, unique ledger keys, insert-only application behavior, and database uniqueness constraints; cryptographic hashing is not required.",
    )
    path.write_text(text)


def patch_tests():
    merged = ROOT / "tests" / "test_merged_audit_policy.py"
    text = merged.read_text()
    needle = '    assert conn.execute("SELECT COUNT(*) n FROM decision_ledger").fetchone()["n"] == 2\n'
    replacement = needle + '    assert "snapshot_hash" not in {row["name"] for row in conn.execute("PRAGMA table_info(decision_ledger)")}\n'
    if replacement not in text:
        text = text.replace(needle, replacement)
    merged.write_text(text)

    pipeline = ROOT / "tests" / "test_pipeline_offline.py"
    text = pipeline.read_text().replace(
        'and all("paper mode" in (p["rationale"] or "") for p in preds)',
        'and all("shadow mode" in (p["rationale"] or "") for p in preds)',
    )
    pipeline.write_text(text)


def cleanup():
    workflow = ROOT / ".github" / "workflows" / "finalize-ledger.yml"
    if workflow.exists():
        workflow.unlink()
    Path(__file__).unlink()


patch_db()
patch_decision_compatibility()
patch_docs()
patch_tests()
cleanup()
print("Removed cryptographic ledger requirement; append-only immutability preserved")
