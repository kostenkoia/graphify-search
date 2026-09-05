import hashlib
from pathlib import Path

import pytest

from benchmark.harness import freeze_model


def test_freeze_hashes_files_and_records_relative_links(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "blob.bin").write_bytes(b"payload")
    (src / "real_dir").mkdir()
    (src / "real_dir" / "nested.txt").write_text("x")
    (src / "link_to_file.bin").symlink_to("blob.bin")
    (src / "link_to_dir").symlink_to("real_dir", target_is_directory=True)

    dst = tmp_path / "dst"
    result = freeze_model.freeze(src, dst)

    assert result["files"] == {
        "blob.bin": hashlib.sha256(b"payload").hexdigest(),
        "real_dir/nested.txt": hashlib.sha256(b"x").hexdigest(),
    }
    assert result["links"] == {"link_to_file.bin": "blob.bin", "link_to_dir": "real_dir"}


def test_freeze_rejects_a_dangling_symlink(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "dangling").symlink_to("does_not_exist")

    with pytest.raises(SystemExit, match="does not resolve"):
        freeze_model.freeze(src, tmp_path / "dst")


def test_freeze_rejects_a_symlink_that_escapes_dst(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"x")
    (src / "escapee").symlink_to(outside)

    with pytest.raises(SystemExit, match="escapes"):
        freeze_model.freeze(src, tmp_path / "dst")


def test_freeze_refuses_to_overwrite_an_existing_dst(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    dst = tmp_path / "dst"
    dst.mkdir()

    with pytest.raises(SystemExit, match="exists"):
        freeze_model.freeze(src, dst)
