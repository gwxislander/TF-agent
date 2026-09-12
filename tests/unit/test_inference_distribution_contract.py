from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_distribution_document_points_to_repository_with_fd_fix():
    """新机器按文档克隆时，必须拿到已包含 FD 修复的维护仓库。"""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "https://github.com/KD-CHL/TF-agent.git" in readme
    assert "https://github.com/KD-CHL/TF-agent/releases" in readme
    assert "https://github.com/gwxislander/TF-agent.git" not in readme


def test_batch_inference_does_not_enable_persistent_workers():
    """批量逐景创建 DataLoader 时不得重新引入 macOS worker/socket 泄漏。"""
    source = (ROOT / "TF-agent" / "pre_engine.py").read_text(encoding="utf-8")

    assert "NUM_WORKERS = 0" in source
    assert "persistent_workers=True" not in source
    assert "try:\n            for raw_batch" in source
    assert "finally:" in source
