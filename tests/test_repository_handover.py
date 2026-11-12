"""Public checkout hygiene; no GPU, training data or running simulator needed."""

from pathlib import Path
import ast
import re
import subprocess
from urllib.parse import urlsplit

import pytest


ROOT = Path(__file__).resolve().parents[1]
KCF = ROOT / "KCF_MPC"


@pytest.mark.parametrize("document", [
    ROOT / "README.md", KCF / "README.md",
    KCF / "docs" / "HANDOVER.md", KCF / "docs" / "finetuning_summary.md",
    KCF / "docs" / "stand_walk_boundary.md", KCF / "docs" / "mpc_method_decision.md",
    KCF / "scripts" / "README.md", KCF / "docs" / "cleanup_verification.md",
])
def test_public_documentation_has_no_broken_local_links(document):
    for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", document.read_text()):
        parsed = urlsplit(target)
        if parsed.scheme or not parsed.path:
            continue
        assert (document.parent / parsed.path).exists(), (document, target)


@pytest.mark.parametrize("repository,path,ignored", [
    (ROOT, "tests/new_test.py", False),
    (ROOT, "data/new_plot.py", False),
    (ROOT, "data/data/example/g1_rom_data.npz", True),
    (ROOT, "data/finetune_collection_v1/manifest.json", True),
    (KCF, "tests/new_test.py", False),
    (KCF, "docs/new_guide.md", False),
    (KCF, "results/g1_kcf_handoff_epoch01540/model_final.pt", False),
    (KCF, "results/g1_kcf_handoff_epoch01540/normalization.npz", False),
    (KCF, "results/g1_kcf_handoff_epoch01540/evaluation/plot.png", True),
    (KCF, "results/rejected_candidate/model_final.pt", True),
    (KCF, "data/processed/train.npz", True),
])
def test_ignore_rules_separate_source_and_selected_model_from_generated_artifacts(
    repository, path, ignored,
):
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--quiet", path],
        cwd=repository, check=False,
    )
    assert result.returncode in (0, 1)
    assert (result.returncode == 0) == ignored


def test_submodule_can_be_cloned_without_private_ssh_credentials():
    result = subprocess.run(
        ["git", "config", "-f", ".gitmodules", "--get", "submodule.KCF_MPC.url"],
        cwd=ROOT, check=True, text=True, capture_output=True,
    )
    assert result.stdout.strip() == "https://github.com/StellaKombo/KCF_MPC.git"


def test_controller_library_is_separate_from_simulation_entrypoints():
    assert {path.name for path in (KCF / "mpc").glob("*.py")} == {
        "mpc.py", "projected_jax.py", "jax_networks.py", "tracking.py",
        "single_integrator.py", "mpc_buffered.py", "mpc_handover.py",
        "mpc_runtime.py", "mpc_trace.py",
    }
    assert {path.name for path in (KCF / "scripts").glob("*.py")} == {
        "visual_recovery_trial.py", "headless_validation.py",
        "benchmark_runtime.py", "replay_live.py",
    }
    for path in (KCF / "mpc").glob("*.py"):
        source = path.read_text()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("KCF_MPC.scripts"), path
            elif isinstance(node, ast.Import):
                assert not any(alias.name.startswith("KCF_MPC.scripts") for alias in node.names), path
        assert "ArgumentParser" not in source, path


@pytest.mark.parametrize("retired", [
    "train.py", "train_projected.py", "data.py", "process.py", "evaluate.py",
    "mpc/mpc_jax.py", "scripts/run_projected_sweep.py",
])
def test_retired_offline_and_legacy_entrypoints_are_not_shipped(retired):
    assert not (KCF / retired).exists()
