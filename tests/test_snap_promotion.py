"""Run the real promotion shell with isolated, stateful Store/attestation doubles."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/snap-promote.yml"
SOURCE_DIGEST = "a" * 40

# Every network-capable executable used by the step is replaced. Unexpected
# commands fail rather than falling through to a locally authenticated CLI.
COMMAND_DOUBLE = r'''
import json
import os
from pathlib import Path
import sys

root = Path(os.environ["PROMOTION_FIXTURE"])
state_path = root / "state.json"
state = json.loads(state_path.read_text())
tool = Path(sys.argv[0]).name
arguments = sys.argv[1:]
with (root / "commands.jsonl").open("a") as stream:
    stream.write(json.dumps([tool, *arguments]) + "\n")

def save():
    state_path.write_text(json.dumps(state))

scenario = state["scenario"]
if tool == "snapcraft" and arguments[:2] == ["revisions", "nvbroadcast"]:
    architecture = arguments[-1]
    if scenario == "other_track":
        print(998, "2026-11-02", architecture, "7.8.9", "otherlatest/candidate*,otherlatest/" + state["channel"] + "*")
    rows = {}
    for channel, revision in (("candidate", state["candidate"][architecture]),
                              (state["channel"], state["target"].get(architecture))):
        if revision is not None:
            rows.setdefault(revision, []).append("latest/" + channel + "*")
    for revision, channels in rows.items():
        version = "7.8.9" if revision in (191, 192, 193, 194) else "7.8.8"
        found_architecture = architecture
        if "latest/candidate*" in channels:
            if scenario == "stale_version": version = "7.8.8"
            if scenario == "wrong_architecture": found_architecture = "s390x"
        print(revision, "2026-11-02", found_architecture, version, ",".join(channels))
elif tool == "snap" and arguments[:2] == ["download", "nvbroadcast"]:
    if scenario == "download_failure": sys.exit(1)
    revision = next(x.split("=", 1)[1] for x in arguments if x.startswith("--revision="))
    directory = Path(next(x.split("=", 1)[1] for x in arguments if x.startswith("--target-directory=")))
    (directory / ("nvbroadcast_" + revision + ".snap")).write_text("fixture " + revision)
elif tool == "gh" and arguments[:2] == ["attestation", "verify"]:
    revision = int(Path(arguments[2]).stem.split("_")[-1])
    if scenario == "attestation_failure" and revision == 191: sys.exit(1)
    if scenario == "candidate_changed" and revision == 191:
        state["candidate"] = {"amd64": 193, "arm64": 194}
        save()
elif tool == "snapcraft" and arguments[:2] == ["release", "nvbroadcast"]:
    revision = int(arguments[2])
    assert arguments[3] == state["channel"], arguments
    architecture = "amd64" if revision in (185, 192, 193) else "arm64"
    if scenario == "release_failure" and revision == 192: sys.exit(1)
    if scenario == "external_publisher" and revision == 192:
        state["target"]["arm64"] = 999
        save()
        sys.exit(1)
    state["target"][architecture] = revision
    if scenario == "candidate_race" and revision == 191:
        state["candidate"] = {"amd64": 193, "arm64": 194}
    save()
    if scenario == "lost_response" and revision == 192: sys.exit(1)
elif tool == "snapcraft" and arguments[0] == "upload-metadata":
    assert state["channel"] == "stable", arguments
else:
    raise AssertionError((tool, arguments))
'''


class SnapPromotionTests(unittest.TestCase):
    def run_promotion(self, channel="edge", scenario="success", **changes):
        workflow = WORKFLOW.read_text()
        step = workflow.split("      - name: Promote tested candidate build set to edge or stable\n", 1)[1]
        script = textwrap.dedent(step.split("        run: |\n", 1)[1])
        with tempfile.TemporaryDirectory(prefix="snap-promotion-") as directory:
            root = Path(directory)
            (root / "snap").mkdir()
            (root / "snap/snapcraft.yaml").write_text("name: nvbroadcast\nversion: '7.8.9'\n")
            binary_directory = root / "bin"
            binary_directory.mkdir()
            for name in ("snapcraft", "snap", "gh", "git"):
                path = binary_directory / name
                path.write_text(f"#!{sys.executable}\n" + COMMAND_DOUBLE)
                path.chmod(0o755)
            state = {
                "channel": channel, "scenario": scenario,
                "candidate": {"amd64": 192, "arm64": 191},
                "target": {"amd64": 185, "arm64": 184},
            }
            if scenario == "empty_target": state["target"].pop("arm64")
            if scenario == "already_promoted": state["target"] = dict(state["candidate"])
            (root / "state.json").write_text(json.dumps(state))
            environment = {
                "PATH": str(binary_directory) + os.pathsep + os.defpath,
                "PROMOTION_FIXTURE": str(root), "RUNNER_TEMP": str(root),
                "TARGET_CHANNEL": channel, "EDGE_PROMOTION_ENABLED": "true",
                "AMD64_REVISION": "", "ARM64_REVISION": "",
                "SNAPCRAFT_STORE_CREDENTIALS": "fixture-only-not-a-credential",
                "GITHUB_SHA": SOURCE_DIGEST,
            }
            environment.update(changes)
            result = subprocess.run(
                ["bash", "-c", script], cwd=root, env=environment,
                capture_output=True, text=True, timeout=20,
            )
            log = root / "commands.jsonl"
            commands = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
            return result, commands, json.loads((root / "state.json").read_text())

    def mutations(self, commands):
        return [x for x in commands if x[:2] in (
            ["snapcraft", "release"], ["snapcraft", "upload-metadata"], ["snapcraft", "promote"],
        )]

    def test_disabled_edge_and_missing_credentials_stop_before_any_store_command(self):
        for channel, changes in (
            ("edge", {"EDGE_PROMOTION_ENABLED": ""}),
            ("edge", {"EDGE_PROMOTION_ENABLED": "false"}),
            ("edge", {"SNAPCRAFT_STORE_CREDENTIALS": ""}),
            ("stable", {"SNAPCRAFT_STORE_CREDENTIALS": ""}),
        ):
            with self.subTest(channel=channel, changes=changes):
                result, commands, _ = self.run_promotion(channel, **changes)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(commands, [])

    def test_invalid_target_or_explicit_revision_stops_before_store_access(self):
        for channel, changes in (("beta", {}), ("edge", {"AMD64_REVISION": "193"}),
                                 ("stable", {"ARM64_REVISION": "194"})):
            with self.subTest(channel=channel, changes=changes):
                result, commands, _ = self.run_promotion(channel, **changes)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(commands, [])

    def test_stale_metadata_download_and_attestation_failures_cannot_publish(self):
        for channel in ("edge", "stable"):
            for scenario in ("stale_version", "wrong_architecture", "download_failure", "attestation_failure"):
                with self.subTest(channel=channel, scenario=scenario):
                    result, commands, state = self.run_promotion(channel, scenario)
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertEqual(self.mutations(commands), [])
                    self.assertEqual(state["target"], {"amd64": 185, "arm64": 184})

    def test_both_artifacts_are_bound_to_release_source_before_exact_releases(self):
        for channel in ("edge", "stable"):
            with self.subTest(channel=channel):
                result, commands, state = self.run_promotion(channel)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                attestations = [x for x in commands if x[:3] == ["gh", "attestation", "verify"]]
                self.assertEqual(len(attestations), 2)
                for command in attestations:
                    self.assertEqual(command[command.index("--source-ref") + 1], "refs/tags/v7.8.9")
                    self.assertEqual(command[command.index("--source-digest") + 1], SOURCE_DIGEST)
                    self.assertEqual(command[command.index("--repo") + 1], "Hkshoonya/nvidia-broadcast-linux")
                    self.assertEqual(command[command.index("--signer-workflow") + 1],
                                     "Hkshoonya/nvidia-broadcast-linux/.github/workflows/snap.yml")
                    self.assertIn("--deny-self-hosted-runners", command)
                mutations = self.mutations(commands)
                self.assertEqual(mutations[:2], [
                    ["snapcraft", "release", "nvbroadcast", "191", channel],
                    ["snapcraft", "release", "nvbroadcast", "192", channel],
                ])
                self.assertLess(commands.index(attestations[-1]), commands.index(mutations[0]))
                self.assertEqual(state["target"], {"amd64": 192, "arm64": 191})
                self.assertEqual(len(mutations), 3 if channel == "stable" else 2)
                if channel == "stable":
                    self.assertEqual(mutations[-1][:2], ["snapcraft", "upload-metadata"])
                    self.assertTrue(mutations[-1][2].endswith("nvbroadcast_191.snap"))

    def test_candidate_change_during_attestation_stops_before_first_release(self):
        result, commands, state = self.run_promotion(scenario="candidate_changed")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Candidate changed during verification", result.stdout)
        self.assertEqual(self.mutations(commands), [])
        self.assertEqual(state["target"], {"amd64": 185, "arm64": 184})

    def test_channel_race_after_freshness_check_cannot_substitute_unverified_revisions(self):
        result, commands, state = self.run_promotion(scenario="candidate_race")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.mutations(commands), [
            ["snapcraft", "release", "nvbroadcast", "191", "edge"],
            ["snapcraft", "release", "nvbroadcast", "192", "edge"],
        ])
        self.assertEqual(state["candidate"], {"amd64": 193, "arm64": 194})
        self.assertEqual(state["target"], {"amd64": 192, "arm64": 191})

    def test_partial_release_and_lost_response_restore_previous_revisions(self):
        for channel in ("edge", "stable"):
            for scenario in ("release_failure", "lost_response"):
                with self.subTest(channel=channel, scenario=scenario):
                    result, commands, state = self.run_promotion(channel, scenario)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(state["target"], {"amd64": 185, "arm64": 184})
                    self.assertFalse(any(x[:2] == ["snapcraft", "upload-metadata"] for x in commands))

    def test_rollback_leaves_another_publishers_revision_unchanged(self):
        result, commands, state = self.run_promotion(scenario="external_publisher")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Another publisher changed arm64 edge", result.stdout)
        self.assertEqual(state["target"], {"amd64": 185, "arm64": 999})
        self.assertNotIn(["snapcraft", "release", "nvbroadcast", "184", "edge"], commands)

    def test_empty_target_refuses_partial_set_without_rollback_baseline(self):
        result, commands, _ = self.run_promotion(scenario="empty_target")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("required for rollback", result.stdout)
        self.assertEqual(self.mutations(commands), [])

    def test_already_promoted_set_is_verified_without_redundant_releases(self):
        result, commands, _ = self.run_promotion(scenario="already_promoted")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len([x for x in commands if x[:3] == ["gh", "attestation", "verify"]]), 2)
        self.assertEqual(self.mutations(commands), [])

    def test_only_exact_latest_track_channel_markers_select_revisions(self):
        result, commands, state = self.run_promotion(scenario="other_track")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.mutations(commands), [
            ["snapcraft", "release", "nvbroadcast", "191", "edge"],
            ["snapcraft", "release", "nvbroadcast", "192", "edge"],
        ])
        self.assertEqual(state["target"], {"amd64": 192, "arm64": 191})


if __name__ == "__main__":
    unittest.main()
