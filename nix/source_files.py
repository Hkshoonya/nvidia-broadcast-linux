"""Update/check the explicit Nix source set from the trusted Git index."""

import argparse
from pathlib import Path
import subprocess
import shlex


def tracked_source(root: Path) -> list[str]:
    data = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
    paths = [x.decode("utf-8") for x in data.split(b"\0") if x]
    if any("\n" in path or "\r" in path for path in paths):
        raise ValueError("Nix source manifest does not support control characters in paths")
    return sorted(path for path in paths
                  if path not in {"flake.nix", "flake.lock"} and not path.startswith("nix/"))


def check_patches(root: Path) -> list[dict]:
    """Check the recipe's literal application substitutions before compilation."""
    recipe = (root / "nix/package.nix").read_text()
    block = recipe.split("\n  postPatch = ''\n", 1)[1].split("\n  '';", 1)[0]
    tokens = shlex.split(block.replace("\\\n", " "), comments=True)
    target, index, records = None, 0, []
    while index < len(tokens):
        if tokens[index] == "substituteInPlace":
            target, index = tokens[index + 1], index + 2
        elif tokens[index] == "--replace-fail":
            path = root / target
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("Recipe substitution target is outside the source")
            pattern = tokens[index + 1]
            count = path.read_text().count(pattern)
            if not count:
                raise ValueError("Obsolete Nix source substitution in " + target)
            records.append({"file": target, "pattern": pattern, "matches": count})
            index += 3
        else:
            index += 1
    if not records:
        raise ValueError("No application source substitutions were checked")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    expected = "\n".join(tracked_source(root)) + "\n"
    target = root / "nix/source-files.txt"
    if arguments.check:
        if target.read_text() != expected:
            parser.exit(1, "Nix source list differs from Git; run python3 nix/source_files.py after staging intended files.\n")
        check_patches(root)
    else:
        target.write_text(expected)
