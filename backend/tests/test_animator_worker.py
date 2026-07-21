"""
Regression test for issue #5 — missing musetalk_worker.py.

musetalk_worker.py is our custom persistent-worker driver, not part of the
upstream MuseTalk repo that setup_musetalk.sh clones. It's gitignored under
models/, so a fresh clone never had it and MuseTalk silently fell back to the
static-image engine for every user. We now ship a tracked copy at
backend/musetalk_worker.py and resolve to it when the clone lacks the script.
"""

import asyncio
import json
from pathlib import Path

import pytest

from app.services.animator import AvatarAnimator


def test_tracked_worker_script_is_shipped():
    """The tracked worker must exist in the repo (outside gitignored models/)."""
    tracked = Path(__file__).resolve().parent.parent / "musetalk_worker.py"
    assert tracked.is_file(), "backend/musetalk_worker.py is missing — MuseTalk will not start"


def test_worker_disables_autograd_during_inference():
    """Persistent turns must not retain autograd graphs and exhaust VRAM."""
    tracked = Path(__file__).resolve().parent.parent / "musetalk_worker.py"
    source = tracked.read_text()
    assert "@torch.inference_mode()\ndef _run_job" in source


def test_worker_preserves_generated_video_when_muxing_audio():
    """Adding audio must not apply a second lossy H.264 encode."""
    tracked = Path(__file__).resolve().parent.parent / "musetalk_worker.py"
    source = tracked.read_text()
    assert 'f"-map 1:v:0 -map 0:a:0 -c:v copy ' in source


def test_resolve_worker_prefers_clone_then_falls_back(tmp_path):
    animator = AvatarAnimator()

    # Clone has its own copy → use it.
    clone = tmp_path / "MuseTalk"
    (clone / "scripts").mkdir(parents=True)
    in_clone = clone / "scripts" / "musetalk_worker.py"
    in_clone.write_text("# worker")
    assert animator._resolve_worker_script(clone) == in_clone

    # Clone lacks the script → fall back to the tracked backend copy.
    empty_clone = tmp_path / "EmptyClone"
    (empty_clone / "scripts").mkdir(parents=True)
    resolved = animator._resolve_worker_script(empty_clone)
    assert resolved.name == "musetalk_worker.py"
    assert resolved.is_file()
    # It's the tracked backend copy, not anything under the empty clone.
    assert "EmptyClone" not in str(resolved)


@pytest.mark.asyncio
async def test_worker_ready_ignores_upstream_stdout_chatter():
    animator = AvatarAnimator()
    reader = asyncio.StreamReader()
    reader.feed_data(b"load vae model\nload unet model\nREADY\n")
    reader.feed_eof()
    proc = type("Proc", (), {"stdout": reader})()

    await animator._wait_for_worker_ready(proc)


@pytest.mark.asyncio
async def test_worker_result_ignores_upstream_stdout_chatter():
    animator = AvatarAnimator()
    reader = asyncio.StreamReader()
    expected = {"status": "ok", "output": "/tmp/video.mp4"}
    reader.feed_data(b"reading images...\n")
    reader.feed_data((json.dumps(expected) + "\n").encode())
    reader.feed_eof()
    proc = type("Proc", (), {"stdout": reader})()

    assert await animator._wait_for_worker_result(proc) == expected
