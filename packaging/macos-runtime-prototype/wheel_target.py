"""Rank wheels for the declared deployment floor, independently of builder OS."""

import json
import sys

from packaging.tags import compatible_tags, cpython_tags, mac_platforms
from packaging.utils import parse_wheel_filename


def rank_wheels(names: list[str]) -> dict[str, int | None]:
    platforms = list(mac_platforms(version=(13, 0), arch="arm64"))
    target_tags = [*cpython_tags(python_version=(3, 13), abis=["cp313"], platforms=platforms),
                   *compatible_tags(python_version=(3, 13), interpreter="cp313", platforms=platforms)]
    ranks = {tag: index for index, tag in reversed(list(enumerate(target_tags)))}
    return {name: min((ranks[tag] for tag in parse_wheel_filename(name)[3] if tag in ranks), default=None)
            for name in names}


if __name__ == "__main__":
    print(json.dumps(rank_wheels(json.load(sys.stdin))))
