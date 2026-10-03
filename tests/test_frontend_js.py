# -*- coding: utf-8 -*-
"""JavaScript-level tests: syntax + headless router execution.

Skipped (not failed) when Node.js is unavailable.
"""

import shutil
import subprocess

import pytest

from pathlib import Path

BACKEND_DIR = Path("D:\\HARSH\\projects\\Timepass\\NLP")
JS = BACKEND_DIR / "frontend" / "nlp-integration.js"
HARNESS = BACKEND_DIR / "tests" / "router_harness.js"

needs_node = pytest.mark.skipif(
    shutil.which("node") is None, reason="Node.js not available"
)


@needs_node
def test_integration_js_syntax():
    proc = subprocess.run(
        ["node", "--check", str(JS)], capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, proc.stderr


@needs_node
def test_router_boot_harness():
    proc = subprocess.run(
        ["node", str(HARNESS), str(JS)],
        capture_output=True, text=True, timeout=60,
        cwd=str(BACKEND_DIR),
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "ROUTER HARNESS OK" in proc.stdout
