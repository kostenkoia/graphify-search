import subprocess
import sys

import pytest

from benchmark.harness import __main__ as cli

VERBS = {"run", "attempt", "abort", "audit", "expand", "freeze-model", "build-symbols", "seal",
        "questions", "summary", "report"}


def test_every_verb_is_registered():
    assert set(cli.VERBS) == VERBS


def test_unknown_verb_exits_two(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["no-such-verb"])
    assert exc.value.code == 2
    assert "no-such-verb" in capsys.readouterr().err


def test_no_verb_lists_them(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert all(v in err for v in VERBS)


@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_help_lists_verbs_and_exits_zero(capsys, flag):
    assert cli.main([flag]) == 0
    out = capsys.readouterr().out
    assert all(v in out for v in VERBS)


@pytest.mark.parametrize("verb", sorted(VERBS))
def test_each_verb_answers_help(verb):
    proc = subprocess.run([sys.executable, "-m", "benchmark.harness", verb, "--help"],
                          capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    first = proc.stdout.splitlines()[0]
    assert first.startswith("usage:"), first
    assert f"benchmark.harness {verb}" in first, first
