"""Pure functions the harness and the audit both run."""

from __future__ import annotations

import base64
import codecs
import csv
import hashlib
import json
import os
from pathlib import Path

import yaml

# NOT DERIVED: read size while hashing; any value yields the same digest
_CHUNK = 1 << 20
_ROOT = Path(__file__).resolve().parents[1]


def _euid() -> int:
    return os.geteuid()


def _owner(path: Path) -> int:
    return path.stat().st_uid


def _mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def sha256_file(path: Path) -> str:
    """Return the hex sha256 of a file's bytes; a symlink hashes its target string."""
    if path.is_symlink():
        return hashlib.sha256(os.readlink(path).encode("utf-8")).hexdigest()
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _drop(obj: object, dotted: str) -> None:
    head, _, rest = dotted.partition(".")
    if not isinstance(obj, dict) or head not in obj:
        return
    if rest:
        _drop(obj[head], rest)
    else:
        del obj[head]


def canonical_hash(text: str, volatile: list[str]) -> str:
    """Hash a tool reply with its volatile keys removed.

    Parameters
    ----------
    text : str
        The `.out` text. JSON is normalised; anything else is hashed as bytes.
    volatile : list of str
        Dotted key paths to delete before hashing.
    """
    stripped = text.strip()
    if stripped.startswith(("{", "[")):
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            obj = None
        if obj is not None:
            for key in volatile:
                _drop(obj, key)
            payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            return hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_hash_file(path: Path, volatile: list[str]) -> str:
    """Hash a reply on disk exactly as `canonical_hash` would, without holding it in memory.

    Parameters
    ----------
    path : Path
        The `.out` file.
    volatile : list of str
        Dotted key paths to delete before hashing, for a JSON reply.

    Returns
    -------
    str
        The same digest `canonical_hash` returns for the file's decoded text.
    """
    with path.open("rb") as fh:
        # inv: the scan decodes and strips as str, because str.strip()'s whitespace class is wider
        # than bytes.lstrip()'s and the two hash forms must agree on every input
        scout = codecs.getincrementaldecoder("utf-8")("replace")
        head = ""
        while not head:
            chunk = fh.read(_CHUNK)
            # inv: reading on until a non-space character appears matches str.strip() for any
            # amount of leading space, however many reads it spans
            head = (scout.decode(chunk) if chunk else scout.decode(b"", final=True)).lstrip()
            if not chunk:
                break
        # why: dropping volatile keys needs the JSON parsed whole, so only the plain-text case streams
        if head.startswith(("{", "[")):
            return canonical_hash(path.read_bytes().decode("utf-8", errors="replace"), volatile)
        fh.seek(0)
        # inv: an incremental decoder, because a multi-byte character straddling two chunks would
        # otherwise decode to two replacements and change the digest
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        digest = hashlib.sha256()
        while chunk := fh.read(_CHUNK):
            digest.update(decoder.decode(chunk).encode("utf-8"))
        digest.update(decoder.decode(b"", final=True).encode("utf-8"))
        return digest.hexdigest()


def listing(root: Path) -> dict[str, tuple[int, int]]:
    """Map every file under `root` to (size, mtime_ns), without following symlinks.

    Parameters
    ----------
    root : Path
        Directory to walk.

    Returns
    -------
    dict of str to tuple of int
        POSIX-relative path to (size, mtime_ns) for every file and every
        symlink; a symlinked directory is recorded but not descended into.

    Raises
    ------
    NotADirectoryError
        If `root` does not exist or is not a directory.
    """
    if not root.is_dir():
        raise NotADirectoryError(root)
    out: dict[str, tuple[int, int]] = {}
    for dirpath, dirs, files in os.walk(root, followlinks=False):
        for name in dirs:
            p = Path(dirpath) / name
            if p.is_symlink():
                st = p.lstat()
                out[p.relative_to(root).as_posix()] = (st.st_size, st.st_mtime_ns)
        for name in files:
            p = Path(dirpath) / name
            st = p.lstat()
            out[p.relative_to(root).as_posix()] = (st.st_size, st.st_mtime_ns)
    return out


def labelled_listing(roots: dict[str, Path]) -> dict[str, tuple[int, int]]:
    """Map every file under each labelled root to (size, mtime_ns), keyed `<label>/<rel>`.

    Parameters
    ----------
    roots : dict of str to Path
        Label to root directory.

    Returns
    -------
    dict of str to tuple of int
        `"<label>/<relative path>"` to (size, mtime_ns) for every file under
        a root that exists; a root that does not exist is skipped.

    Raises
    ------
    NotADirectoryError
        If a root exists but is not a directory.
    """
    out: dict[str, tuple[int, int]] = {}
    for label, root in roots.items():
        if not root.exists():
            continue
        # inv: a root that exists but is not a directory (a sandbox clobbered into a file) raises
        # rather than listing as empty, so "nothing here" never stands in for "I could not look"
        for rel, stat in listing(root).items():
            out[f"{label}/{rel}"] = stat
    return out


def changed_files(
    before: dict[str, tuple[int, int]], after: dict[str, tuple[int, int]], roots: dict[str, Path],
) -> list[dict[str, str | None]]:
    """List labelled files whose (size, mtime) differ between two listings, hashed on change.

    Parameters
    ----------
    before, after : dict of str to tuple of int
        Listings keyed `"<label>/<relative path>"`, as returned by `labelled_listing`.
    roots : dict of str to Path
        Label to root directory, used to resolve a changed key back to a file to hash.
    """
    # inv: a rewrite with identical size and mtime_ns is not detected; the harness accepts this.
    changed: list[dict[str, str | None]] = []
    for rel, stat in sorted(after.items()):
        if before.get(rel) != stat:
            label, _, inner = rel.partition("/")
            changed.append({"path": rel, "sha256": sha256_file(roots[label] / inner)})
    for rel in sorted(set(before) - set(after)):
        changed.append({"path": rel, "sha256": None})
    return changed


def _record_rows(site: Path) -> list[tuple[str, Path, str]]:
    rows: list[tuple[str, Path, str]] = []
    for record in sorted(site.glob("*.dist-info/RECORD")):
        with record.open(newline="", encoding="utf-8") as fh:
            for row in csv.reader(fh):
                if len(row) < 2 or not row[1].startswith("sha256="):
                    continue
                path = Path(os.path.normpath(site / row[0]))
                rows.append((os.path.relpath(path, site), path, row[1][len("sha256=") :]))
    return rows


def _urlsafe_sha256(path: Path) -> str:
    raw = hashlib.sha256(path.read_bytes()).digest()
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def verify_records(site: Path) -> list[str]:
    """Return RECORD-listed files whose sha256 no longer matches."""
    return [
        rel for rel, path, expected in _record_rows(site) if not path.is_file() or _urlsafe_sha256(path) != expected
    ]


def environment_hash(site: Path) -> str:
    """Hash the (path, sha256) pairs of every RECORD-listed file under `site`."""
    h = hashlib.sha256()
    for rel, path, _expected in sorted(_record_rows(site)):
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(_urlsafe_sha256(path).encode("ascii") if path.is_file() else b"missing")
        h.update(b"\n")
    return h.hexdigest()


def resolve_launcher(invocation: dict, argv: list[str]) -> list[str]:
    """Replace an `argv[0]` equal to the launcher's basename with the launcher path."""
    pkg = invocation["package"]
    launcher = str(pkg["launcher"])
    if argv and argv[0] == Path(launcher).name:
        return [launcher, *argv[1:]]
    return list(argv)


def _printable(tokens: list[str]) -> str | None:
    for tok in tokens:
        if not tok.isprintable():
            return f"argument is not printable: {tok!r}"
    return None


def _check_script(invocation: dict, script: str) -> str | None:
    allowed: dict[str, str] = invocation.get("allowed_scripts") or {}
    # inv: a bare-basename match would admit the vetted content from any directory the allowlist never named
    hit = next((k for k in allowed if script == k or script.endswith("/" + k)), None)
    if hit is None:
        return f"script not allowed: {script}"
    path = Path(script)
    if not path.is_file():
        return f"script missing: {script}"
    if sha256_file(path) != allowed[hit]:
        return f"script hash differs: {hit}"
    return None


def _check_flags(row: dict, tail: list[str]) -> str | None:
    flags: dict = row.get("flags") or {}
    i = 0
    while i < len(tail):
        tok = tail[i]
        name, has_eq, value = tok.partition("=")
        if name not in flags:
            return f"unconsumed token: {tok}"
        spec = flags[name] or {}
        takes_value = "literal" in spec or "type" in spec
        if takes_value and not has_eq:
            if i + 1 >= len(tail):
                return f"flag needs a value: {name}"
            value, i = tail[i + 1], i + 1
        if takes_value and "literal" in spec and value not in [str(v) for v in spec["literal"]]:
            return f"flag value not allowed: {name}={value}"
        i += 1
    return None


def _check_act(invocation: dict, argv: list[str]) -> str | None:
    if len(argv) < 2:
        return "argv needs a subcommand"
    if (reason := _printable(argv)) is not None:
        return reason
    pkg = invocation["package"]
    head, sub = argv[0], argv[1]
    if head not in (str(pkg["launcher"]), Path(str(pkg["launcher"])).name, str(pkg["interpreter"])):
        return f"argv[0] is not the package: {head}"
    if sub in invocation.get("rejected_subcommands", []):
        return f"subcommand is rejected: {sub}"
    # inv: the interpreter is admitted to run an allowlisted script and nothing else; letting it
    # reach the subcommand table would give a second, unvetted way to invoke the vendor's own verbs
    if head == str(pkg["interpreter"]):
        reason = _check_script(invocation, sub)
        if reason is not None:
            return reason
        # inv: the allowlist vets a script's own content, not any argv a caller could append after it
        return None if len(argv) == 2 else f"unconsumed token: {argv[2]}"
    row = (invocation.get("subcommands") or {}).get(sub)
    if row is None:
        return f"unknown subcommand: {sub}"
    rest = argv[2:]
    rejected = set(row.get("rejected") or [])
    for tok in rest:
        if tok.split("=", 1)[0] in rejected:
            return f"flag is rejected: {tok.split('=', 1)[0]}"
    npos = int(row.get("positional", 0))
    # inv: positionals are consumed first, left to right, because the vendor indexes sys.argv directly
    positionals, tail = rest[:npos], rest[npos:]
    if len(positionals) < npos:
        return f"{sub} needs {npos} positional argument(s)"
    for tok in positionals:
        if tok.startswith("-"):
            return f"positional looks like a flag: {tok}"
    return _check_flags(row, tail)


def _check_tool(invocation: dict, tool: str, args: dict) -> str | None:
    row = (invocation.get("tools") or {}).get(tool)
    if row is None:
        return f"unknown tool: {tool}"
    if (reason := _printable([v for v in args.values() if isinstance(v, str)])) is not None:
        return reason
    keys: dict = row.get("keys") or {}
    for key in args:
        if key in (row.get("rejected") or []):
            return f"key is rejected: {key}"
        if key not in keys:
            return f"unknown key: {key}"
    for key, spec in keys.items():
        if key not in args:
            continue
        spec = spec or {}
        # inv: compared byte-exactly; the vendor treats "Minimal" as a different value than "minimal"
        if "literal" in spec and args[key] not in spec["literal"]:
            return f"key value not allowed: {key}={args[key]!r}"
        if "literal_prefix" in spec and not any(str(args[key]).startswith(p) for p in spec["literal_prefix"]):
            return f"key value lacks the required prefix: {key}"
    return None


def check_call(invocation: dict, call: dict) -> str | None:
    """Return why `call` is refused under `invocation`, or None when it is accepted."""
    if call.get("kind") == "act":
        return _check_act(invocation, list(call["argv"]))
    if call.get("kind") == "tool":
        return _check_tool(invocation, str(call["tool"]), dict(call.get("args") or {}))
    return f"unknown call kind: {call.get('kind')!r}"


# inv: only a record that names a place in the corpus can be an origin; an unparsed line and a
# vendor's empty reply name nothing, so neither may account for an argument
_ORIGIN_KINDS = ("place", "candidate", "edge", "file")
_ORIGIN_FIELDS = ("label", "symbol", "qualified_name", "path")


def provenance(value: object, prompt_text: str, records: list[dict], literals: set) -> dict:
    """Return where one runner argument came from.

    Parameters
    ----------
    value : object
        The argument as the runner sent it.
    prompt_text : str
        The whole prompt the runner was given.
    records : list of dict
        Records of entries earlier than the call being classified.
    literals : set
        Values the vendor's grammar pins for this call.

    Returns
    -------
    dict
        `{"kind": "literal"}`, `{"kind": "prompt"}`, `{"kind": "record", "n": <entry>}`
        or `{"kind": "none"}`.
    """
    # why: the first origin wins, ordered by how early it was available to the runner -- the
    # grammar before the prompt, the prompt before anything an action returned
    if isinstance(value, (str, int, float, bool)) and value in literals:
        return {"kind": "literal"}
    text = value if isinstance(value, str) else str(value)
    if text and prompt_text and text in prompt_text:
        return {"kind": "prompt"}
    if isinstance(value, str):
        for record in records:
            if record.get("kind") not in _ORIGIN_KINDS:
                continue
            # inv: compared byte for byte and never normalised, so two spellings of one glyph
            # stay two different arguments
            if any(record.get(field) == value for field in _ORIGIN_FIELDS):
                return {"kind": "record", "n": record["n"]}
    return {"kind": "none"}


def _act_literals(invocation: dict, argv: list[str]) -> set:
    subcommands = invocation.get("subcommands") or {}
    found: set = set(subcommands)
    row = subcommands.get(argv[1]) if len(argv) > 1 else None
    for name, spec in ((row or {}).get("flags") or {}).items():
        found.add(name)
        found.update((spec or {}).get("literal") or [])
    return found


def _tool_literals(invocation: dict, tool: str) -> set:
    row = (invocation.get("tools") or {}).get(tool) or {}
    found: set = set()
    for spec in (row.get("keys") or {}).values():
        found.update((spec or {}).get("literal") or [])
    return found


def call_provenance(invocation: dict, call: dict, prompt_text: str, records: list[dict]) -> list[dict]:
    """Return the origin of every argument of one runner call.

    Parameters
    ----------
    invocation : dict
        The system's `invocation` block, whose grammar pins the literal values.
    call : dict
        `{"kind": "act", "argv": [...]}` or `{"kind": "tool", "tool": ..., "args": {...}}`.
    prompt_text : str
        The whole prompt the runner was given.
    records : list of dict
        Records of entries earlier than this call.

    Returns
    -------
    list of dict
        One row per argument: where it sits, its value, and its origin.
    """
    if call.get("kind") == "act":
        argv = list(call["argv"])
        literals = _act_literals(invocation, argv)
        # inv: the launcher is written by the harness, not by the runner, so it is not an argument
        return [{"at": f"argv[{i}]", "value": token, **provenance(token, prompt_text, records, literals)}
                for i, token in enumerate(argv) if i > 0]
    args = dict(call.get("args") or {})
    literals = _tool_literals(invocation, str(call["tool"]))
    # inv: `quote` is the runner's citation of the documentation, checked against the frozen
    # document by its own audit, and is not an argument the vendor receives
    return [{"at": key, "value": args[key], **provenance(args[key], prompt_text, records, literals)}
            for key in sorted(args) if key != "quote"]


def check_models(benchmark: Path, system: str, models: dict) -> list[str]:
    """Return every way a system's frozen model directories differ from `harness.yaml`'s `models:`.

    Parameters
    ----------
    benchmark : Path
        The `benchmark/` directory.
    system : str
        The system whose `systems/<system>/models/` is checked.
    models : dict
        `{<dir>: {"files": {<rel>: <sha256>}, "links": {<rel>: <target>}}}`, as `freeze-model` wrote it.

    Returns
    -------
    list of str
        `model differs: <dir>: <rel>` per changed or missing file, `… (link)` per wrong symlink
        target, `… (unlisted)` per file the block does not name; empty when every directory matches.
    """
    problems: list[str] = []
    for name, body in models.items():
        root = benchmark / "systems" / system / "models" / name
        files = body.get("files") or {}
        links = body.get("links") or {}
        for rel, digest in files.items():
            p = root / rel
            if not p.is_file() or sha256_file(p) != digest:
                problems.append(f"model differs: {name}: {rel}")
        for rel, target in links.items():
            p = root / rel
            if not p.is_symlink() or os.readlink(p) != target:
                problems.append(f"model differs: {name}: {rel} (link)")
        listed = set(files) | set(links)
        for p in sorted(x for x in root.rglob("*") if x.is_file() or x.is_symlink()):
            rel = p.relative_to(root).as_posix()
            if rel not in listed:
                problems.append(f"model differs: {name}: {rel} (unlisted)")
    return problems


def check_corpus(snapshot: Path) -> list[str]:
    """Return every file of a snapshot's `source/` that differs from its frozen listing.

    Parameters
    ----------
    snapshot : Path
        The snapshot directory holding `source/`, `fileset.sha256` and, when frozen,
        `symbols.jsonl` and `symbols.sha256`.

    Returns
    -------
    list of str
        `corpus differs: <rel>` per file under `source/` whose hash no longer matches
        `fileset.sha256`, plus `corpus differs: symbols.jsonl` when `symbols.jsonl` no longer
        matches `symbols.sha256`; `corpus is not frozen: no fileset.sha256` when the listing
        itself is missing.
    """
    problems: list[str] = []
    listing = snapshot / "fileset.sha256"
    if not listing.is_file():
        return ["corpus is not frozen: no fileset.sha256"]
    for line in listing.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, _, rel = line.partition("  ")
        rel = rel[2:] if rel.startswith("./") else rel
        p = snapshot / "source" / rel
        if not p.is_file() or sha256_file(p) != digest:
            problems.append(f"corpus differs: {rel}")
    sym = snapshot / "symbols.sha256"
    if sym.is_file():
        digest = sym.read_text(encoding="utf-8").split()[0]
        if not (snapshot / "symbols.jsonl").is_file() or sha256_file(snapshot / "symbols.jsonl") != digest:
            problems.append("corpus differs: symbols.jsonl")
    return problems


def check_instrument(benchmark: Path) -> dict:
    """Check the instrument is sealed and return `machine.yaml`, without the bench-uid gate.

    The shared prologue of `require_sealed`: this is the part of the sealed check that does
    not depend on who is calling. `run` calls this directly -- it runs as the operator, not
    bench, and would always fail the uid gate `require_sealed` makes for real once it is bench,
    over sudo, inside `attempt`.

    Parameters
    ----------
    benchmark : Path
        The `benchmark/` directory a caller was given; must be the one this package lives in.

    Returns
    -------
    dict
        The loaded `machine.yaml`.

    Raises
    ------
    SystemExit
        When `benchmark` is not this harness's own root, the instrument is unlocked or
        unsealed, or the lock is not installed on this machine.
    """
    # why: seal imports rules for hashing, so the import lives here to keep the modules acyclic at load
    from benchmark.harness import seal

    if benchmark.resolve() != _ROOT.resolve():
        raise SystemExit(f"{benchmark} is not the benchmark this harness lives in ({_ROOT})")
    # why: checked before seal.check so an installed machine mid-unlock, owner-writable
    # everywhere by design, refuses in one line instead of behind seal's ownership wall
    if (benchmark / seal.UNLOCKED).exists():
        raise SystemExit("runs are refused while the instrument is unlocked")
    problems = seal.check(benchmark)
    if problems:
        raise SystemExit("\n".join(problems))
    machine_path = benchmark / seal.MACHINE
    if not machine_path.is_file():
        raise SystemExit("lock is not installed on this machine; run benchmark/lock/install")
    return yaml.safe_load(machine_path.read_text(encoding="utf-8")) or {}


def require_sealed(benchmark: Path, tmp_root: Path) -> dict:
    """Refuse to run unless the tree is sealed, locked, owned as the seal expects, and run as bench.

    Parameters
    ----------
    benchmark : Path
        The `benchmark/` directory a caller was given; must be the one this package lives in.
    tmp_root : Path
        The tmp root the run intends to use; must be the one `machine.yaml` names.

    Returns
    -------
    dict
        The loaded `machine.yaml`.

    Raises
    ------
    SystemExit
        With one of the nine refusals of the protocol.
    """
    machine = check_instrument(benchmark)
    if _euid() != int(machine["bench_uid"]):
        raise SystemExit(f"the harness runs as bench, not uid {_euid()}")
    if "tmp_root" not in machine:
        raise SystemExit("machine.yaml is missing tmp_root")
    if Path(machine["tmp_root"]).resolve() != tmp_root.resolve():
        raise SystemExit(f"tmp root {tmp_root} is not the one machine.yaml names ({machine['tmp_root']})")
    if tmp_root.exists():
        if _owner(tmp_root) != int(machine["bench_uid"]):
            raise SystemExit(f"record not owned by bench: {tmp_root}")
        if _mode(tmp_root) & 0o022:
            raise SystemExit(f"record writable by others: {tmp_root}")
    return machine


def machine_facts() -> dict:
    """Read `machine.yaml`'s tmp root, then validate the tree fully through `require_sealed`.

    Returns
    -------
    dict
        The machine facts `require_sealed` returns.

    Raises
    ------
    SystemExit
        With one of `require_sealed`'s refusals.
    """
    # why: seal imports rules for hashing, so the import lives here to keep the modules acyclic at load
    from benchmark.harness import seal

    # inv: require_sealed itself validates tmp_root against machine.yaml, so a raw, unchecked
    # read here only bootstraps the argument it needs -- the real check happens inside it
    raw_path = _ROOT / seal.MACHINE
    raw = (yaml.safe_load(raw_path.read_text(encoding="utf-8")) or {}) if raw_path.is_file() else {}
    return require_sealed(_ROOT, Path(raw.get("tmp_root") or ""))
