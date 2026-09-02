"""AGENT-011: durable job ledger and restart reconciliation."""
from __future__ import annotations

import os
import sys
import sqlite3
import json
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

_TF_AGENT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "TF-agent"))
if _TF_AGENT not in sys.path:
    sys.path.insert(0, _TF_AGENT)

from job_store import JobStore, worker_success_is_committable  # noqa: E402


def test_stop_request_wins_over_worker_success_for_commit_gate():
    assert worker_success_is_committable(True, False) is True
    assert worker_success_is_committable(False, False) is False
    assert worker_success_is_committable(True, True) is False


def _claim_job_in_child(db_path):
    return JobStore(db_path).claim("job-cross-process") is not None


def _create_plan_in_child(db_path):
    row = JobStore(db_path).create(
        task="滩涂", kind="workflow", plan_id="plan-cross-process", job_id=f"child-{os.getpid()}"
    )
    return row.job_id


def _seed_running_job_then_exit(db_path):
    """Persist RUNNING, then terminate abruptly to simulate a crashed worker."""
    store = JobStore(db_path)
    store.create(task="滩涂", kind="dl", job_id="job-crashed-process")
    store.claim("job-crashed-process")
    os._exit(0)


def test_job_claim_is_atomic_and_duplicate_start_is_rejected(tmp_path):
    store = JobStore(str(tmp_path / "jobs.sqlite3"))
    store.create(task="滩涂", kind="workflow", job_id="job-1")
    claimed = store.claim("job-1")
    assert claimed is not None
    assert claimed.status == "RUNNING"
    assert claimed.attempt == 1
    assert store.claim("job-1") is None


def test_active_plan_reuses_existing_job_instead_of_creating_duplicate(tmp_path):
    store = JobStore(str(tmp_path / "jobs-plan-dedup.sqlite3"))
    first = store.create(task="滩涂", kind="workflow", plan_id="plan-same", job_id="job-first")
    second = store.create(task="滩涂", kind="workflow", plan_id="plan-same", job_id="job-second")

    assert first.job_id == "job-first"
    assert second.job_id == first.job_id
    assert [row.job_id for row in store.list()] == ["job-first"]


def test_terminal_plan_can_be_explicitly_requeued(tmp_path):
    store = JobStore(str(tmp_path / "jobs-plan-rerun.sqlite3"))
    first = store.create(task="滩涂", kind="workflow", plan_id="plan-rerun", job_id="job-old")
    store.claim(first.job_id)
    store.transition(first.job_id, "SUCCEEDED", progress=100)

    rerun = store.create(task="滩涂", kind="workflow", plan_id="plan-rerun", job_id="job-new")

    assert rerun.job_id == "job-new"
    assert {row.job_id for row in store.list()} == {"job-old", "job-new"}


def test_reusing_job_id_for_different_plan_is_rejected(tmp_path):
    """请求身份冲突不能静默复用旧账本记录。"""
    store = JobStore(str(tmp_path / "jobs-job-id-conflict.sqlite3"))
    store.create(task="滩涂", kind="workflow", plan_id="plan-old", job_id="job-same")

    try:
        store.create(task="滩涂", kind="workflow", plan_id="plan-new", job_id="job-same")
    except ValueError as exc:
        assert "job_id" in str(exc)
    else:
        raise AssertionError("different plan silently reused the existing job_id")

    assert store.get("job-same").plan_id == "plan-old"


def test_reusing_active_plan_for_different_task_kind_is_rejected(tmp_path):
    """活动 plan_id 复用时，task/kind 也必须保持一致。"""
    store = JobStore(str(tmp_path / "jobs-plan-identity.sqlite3"))
    store.create(task="滩涂", kind="workflow", plan_id="plan-same", job_id="job-same")

    try:
        store.create(task="红树林", kind="dl", plan_id="plan-same", job_id="job-other")
    except ValueError as exc:
        assert "plan_id" in str(exc)
    else:
        raise AssertionError("different task/kind silently reused the active plan_id")

    assert len(store.list()) == 1


def test_job_claim_is_atomic_across_processes(tmp_path):
    path = str(tmp_path / "jobs-cross-process.sqlite3")
    store = JobStore(path)
    store.create(task="滩涂", kind="workflow", job_id="job-cross-process")
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=2, mp_context=context) as pool:
        results = list(pool.map(_claim_job_in_child, [path, path]))
    assert sorted(results) == [False, True]
    assert store.get("job-cross-process").attempt == 1


def test_active_plan_dedup_is_atomic_across_processes(tmp_path):
    path = str(tmp_path / "jobs-plan-cross-process.sqlite3")
    store = JobStore(path)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=2, mp_context=context) as pool:
        results = list(pool.map(_create_plan_in_child, [path, path]))

    assert len(set(results)) == 1
    assert len(store.list()) == 1


def test_reconcile_marks_inflight_jobs_interrupted_without_fake_success(tmp_path):
    store = JobStore(str(tmp_path / "jobs.sqlite3"))
    store.create(task="滩涂", kind="dl", job_id="job-2")
    store.claim("job-2")
    recovered = store.reconcile()
    assert [row.job_id for row in recovered] == ["job-2"]
    assert store.get("job-2").status == "INTERRUPTED"
    assert "进程重启" in (store.get("job-2").error_summary or "")


def test_reconcile_after_abrupt_child_process_exit(tmp_path):
    """真实进程边界退出后，重新打开账本仍能识别并中断未完成任务。"""
    path = str(tmp_path / "jobs-crash-recovery.sqlite3")
    context = multiprocessing.get_context("spawn")
    child = context.Process(target=_seed_running_job_then_exit, args=(path,))
    child.start()
    child.join(timeout=20)
    assert child.exitcode == 0

    reopened = JobStore(path)
    assert reopened.get("job-crashed-process").status == "RUNNING"
    recovered = reopened.reconcile()
    assert [row.job_id for row in recovered] == ["job-crashed-process"]
    assert reopened.get("job-crashed-process").status == "INTERRUPTED"


def test_metadata_and_errors_are_sanitized(tmp_path):
    store = JobStore(str(tmp_path / "jobs.sqlite3"))
    row = store.create(
        task="/Users/chl/private/task",
        kind="workflow",
        job_id="job-3",
        metadata={"prompt": "secret prompt", "note": "/Users/chl/private/file.tif", "ok": "visible"},
    )
    row = store.transition("job-3", "FAILED", error="api_key=sk-secret /Users/chl/private/file.tif /private/tmp/worker.tif")
    assert "/Users/" not in row.task
    assert "prompt" not in row.metadata
    assert "/Users/" not in (row.metadata.get("note") or "")
    assert "sk-secret" not in (row.error_summary or "")
    assert "/private/" not in (row.error_summary or "")


def test_job_ledger_redacts_precise_spatial_metadata(tmp_path):
    store = JobStore(str(tmp_path / "jobs.sqlite3"))
    row = store.create(
        task="滩涂",
        kind="workflow",
        job_id="job-spatial",
        metadata={
            "bounds": "left=120.600000, bottom=30.200000, right=121.200000, top=30.900000",
            "crs": "EPSG:4326",
        },
    )
    row = store.transition(
        "job-spatial",
        "FAILED",
        error="bounds: left=120.600000, bottom=30.200000, right=121.200000, top=30.900000",
    )
    payload = json.dumps(row.__dict__, ensure_ascii=False)
    assert "120.600000" not in payload
    assert "EPSG:4326" not in payload
    assert "<spatial-redacted>" in payload


def test_job_store_sanitizes_windows_and_external_volume_paths(tmp_path):
    store = JobStore(str(tmp_path / "jobs.sqlite3"))
    row = store.create(
        task=r"C:\Users\chl\private\task",
        kind="workflow",
        job_id="job-cross-platform-path",
        metadata={"note": "/Volumes/External/private/result.tif"},
    )
    assert r"C:\Users\chl" not in row.task
    assert "/Volumes/" not in (row.metadata.get("note") or "")


def test_corrupt_ledger_is_preserved(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    path.write_bytes(b"not a sqlite database")
    store = JobStore(str(path))
    store.create(task="滩涂", kind="dl", job_id="job-4")
    assert store.get("job-4") is not None
    assert list(tmp_path.glob("jobs.sqlite3.corrupt-*"))


def test_malformed_job_row_is_preserved_before_rebuilding_ledger(tmp_path):
    path = tmp_path / "jobs-row-corrupt.sqlite3"
    store = JobStore(str(path))
    store.create(task="滩涂", kind="dl", job_id="job-row-corrupt")
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE jobs SET metadata=? WHERE job_id=?", ("not-json", "job-row-corrupt"))
        conn.commit()

    reopened = JobStore(str(path))

    assert reopened.list() == []
    assert list(tmp_path.glob("jobs-row-corrupt.sqlite3.corrupt-*"))


def test_corrupt_ledger_is_quarantined_after_sqlite_connection_closes(tmp_path, monkeypatch):
    """The ledger must be closed before Windows attempts to move its DB file."""
    path = tmp_path / "jobs-close-before-quarantine.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.execute("""CREATE TABLE jobs (
            job_id TEXT PRIMARY KEY, task TEXT NOT NULL, kind TEXT NOT NULL,
            plan_id TEXT, status TEXT NOT NULL, progress INTEGER NOT NULL,
            attempt INTEGER NOT NULL, started_at TEXT, updated_at TEXT NOT NULL,
            artifact_ids TEXT NOT NULL, error_summary TEXT, metadata TEXT NOT NULL
        )""")
        conn.execute(
            "INSERT INTO jobs(job_id,task,kind,status,progress,attempt,updated_at,artifact_ids,metadata) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            ("job-corrupt", "task", "kind", "QUEUED", 0, 0, "now", "[]", "not-json"),
        )

    connections = []
    closed_connection_ids = set()

    class TrackingConnection(sqlite3.Connection):
        def close(self):
            closed_connection_ids.add(id(self))
            return super().close()

    def tracked_connect():
        conn = sqlite3.connect(
            path,
            timeout=10,
            isolation_level=None,
            factory=TrackingConnection,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute("PRAGMA journal_mode=WAL")
        connections.append(conn)
        return conn

    original_preserve = JobStore._preserve_corrupt_ledger
    preserve_observations = []

    def assert_closed_before_preserve(self):
        preserve_observations.append(
            all(id(conn) in closed_connection_ids for conn in connections)
        )
        return original_preserve(self)

    monkeypatch.setattr(JobStore, "_connect", staticmethod(tracked_connect))
    monkeypatch.setattr(JobStore, "_preserve_corrupt_ledger", assert_closed_before_preserve)

    JobStore(str(path))

    assert preserve_observations == [True]
    assert all(id(conn) in closed_connection_ids for conn in connections)


def test_corrupt_ledger_backup_does_not_overwrite_same_second(tmp_path, monkeypatch):
    path = tmp_path / "jobs.sqlite3"
    monkeypatch.setattr("job_store.time.strftime", lambda *args, **kwargs: "20260822000000")
    path.write_bytes(b"not a sqlite database")
    JobStore(str(path))
    path.write_bytes(b"not a sqlite database")
    JobStore(str(path))
    backups = sorted(
        path for path in tmp_path.glob("jobs.sqlite3.corrupt-*")
        if not path.name.endswith(("-wal", "-shm"))
    )
    assert len(backups) == 2
    assert backups[0].name != backups[1].name


def test_corrupt_ledger_preserves_wal_sidecar(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    path.write_bytes(b"not a sqlite database")
    (tmp_path / "jobs.sqlite3-wal").write_bytes(b"stale wal")
    JobStore(str(path))
    assert list(tmp_path.glob("jobs.sqlite3.corrupt-*-wal"))


def test_terminal_job_cannot_move_back_to_running(tmp_path):
    store = JobStore(str(tmp_path / "jobs.sqlite3"))
    store.create(task="滩涂", kind="dl", job_id="job-5")
    store.claim("job-5")
    store.transition("job-5", "SUCCEEDED", progress=100)
    try:
        store.transition("job-5", "RUNNING")
    except ValueError as exc:
        assert "invalid job transition" in str(exc)
    else:
        raise AssertionError("terminal job unexpectedly transitioned back to RUNNING")


def test_progress_update_is_durable_without_changing_attempt_or_status(tmp_path):
    store = JobStore(str(tmp_path / "jobs.sqlite3"))
    store.create(task="滩涂", kind="dl", job_id="job-progress")
    running = store.claim("job-progress")
    updated = store.update_progress("job-progress", 47, metadata={"phase": "INFERENCE"})
    assert updated.status == "RUNNING"
    assert updated.progress == 47
    assert updated.attempt == running.attempt == 1
    assert updated.metadata["phase"] == "INFERENCE"

    reopened = JobStore(str(tmp_path / "jobs.sqlite3"))
    persisted = reopened.get("job-progress")
    assert persisted.progress == 47
    assert persisted.status == "RUNNING"


def test_execution_request_audit_metadata_survives_reopen(tmp_path):
    store = JobStore(str(tmp_path / "jobs.sqlite3"))
    row = store.create(
        task="滩涂",
        kind="autotune",
        plan_id="plan-at-1",
        job_id="exec-at-1",
        metadata={
            "confirmation_source": "ui",
            "request_id": "exec-at-1",
            "request_schema": "execution_request_v1",
            "entrypoint": "autotune",
        },
    )
    assert row.metadata["request_schema"] == "execution_request_v1"
    reopened = JobStore(str(tmp_path / "jobs.sqlite3"))
    persisted = reopened.get("exec-at-1")
    assert persisted.metadata["entrypoint"] == "autotune"
    assert persisted.metadata["request_id"] == "exec-at-1"
