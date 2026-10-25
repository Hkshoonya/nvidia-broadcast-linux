"""Regression checks for the manual native-candidate job trust boundaries.

These checks use the deliberately simple, fixed indentation of these workflow
files, rather than introducing a YAML runtime dependency into packaging tests.
Full YAML parsing and shell validation accompany workflow review separately.
"""
from pathlib import Path
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github/workflows"
MAIN = (WORKFLOWS / "native-cpu-finalize.yml").read_text()
CALLED = (WORKFLOWS / "native-cpu-qualify.yml").read_text()
PRODUCER = (WORKFLOWS / "native-cpu-candidate.yml").read_text()


def blocks(text, indent):
    matches = list(re.finditer(r"^" + " " * indent + r"([A-Za-z0-9_-]+):(?:[^\n]*)\n", text, re.MULTILINE))
    return {match.group(1): text[match.start():matches[index + 1].start() if index + 1 < len(matches) else len(text)]
            for index, match in enumerate(matches)}


def jobs(text):
    return blocks(text.split("\njobs:\n", 1)[1], 2)


def steps(job):
    matches = list(re.finditer(r"^      - ", job, re.MULTILINE))
    return [job[match.start():matches[index + 1].start() if index + 1 < len(matches) else len(job)]
            for index, match in enumerate(matches)]


def runs(step):
    match = re.search(r"^        run: \|\n((?:          [^\n]*\n|\n)*)", step, re.MULTILINE)
    return "" if match is None else "\n".join(line[10:] for line in match.group(1).splitlines())


def python_body(step):
    return runs(step).split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]


def copy_policy(root):
    target = root / "packaging/native-runtime/candidate_policy.py"
    target.parent.mkdir(parents=True)
    target.write_bytes((ROOT / "packaging/native-runtime/candidate_policy.py").read_bytes())


class NativeCandidateWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.main = jobs(MAIN)
        self.called = jobs(CALLED)
        self.producer = jobs(PRODUCER)["build-cpu"]
        self.assertEqual(set(self.main), {"validate", "sign", "qualify-deb", "qualify-rpm", "retain"})
        self.assertEqual(set(self.called), {"qualify"})

    def test_fresh_runner_preparation_creates_fetch_destination_after_capacity_check(self):
        step = next(step for step in steps(self.main["validate"]) if "shutil.disk_usage" in runs(step))
        code = python_body(step)
        for variant, free, success in (("cpu", 12 * 1024**3, True), ("cpu", 12 * 1024**3 - 1, False),
                                       ("cuda", 40 * 1024**3, True), ("cuda", 40 * 1024**3 - 1, False),
                                       ("cuda", 13 * 1024**3, False), ("../cuda", 100 * 1024**3, False)):
            with self.subTest(variant=variant, free=free), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                copy_policy(root)
                # Exercise the actual workflow program in a clean directory;
                # only capacity is simulated so the test is host-independent.
                prelude = f"import shutil, types\nshutil.disk_usage = lambda path: types.SimpleNamespace(free={free})\n"
                environment = {**os.environ, "GITHUB_WORKSPACE": temporary, "RUNNER_TEMP": temporary,
                               "CANDIDATE_VARIANT": variant}
                result = subprocess.run([sys.executable, "-I", "-S", "-c", prelude + code],
                                        cwd=root, env=environment, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode == 0, success, result.stderr)
                destination = root / "dist/builder-delivery/builders.tar"
                if success:
                    with destination.open("xb") as stream:
                        stream.write(b"fetch destination ready")
                else:
                    self.assertFalse(destination.parent.exists())

    def test_producer_records_selected_capacity_before_loading_builders(self):
        step = next(s for s in steps(self.producer) if "shutil.disk_usage" in runs(s))
        code = python_body(step)
        for variant, free, expected in (("cpu", 12 * 1024**3, 12 * 1024**3),
                                        ("cuda", 40 * 1024**3, 40 * 1024**3),
                                        ("cuda", 12 * 1024**3, None),
                                        ("cuda; touch injected", 100 * 1024**3, None)):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                copy_policy(root)
                (root / "packaging/native-runtime/builder-delivery.json").write_text(json.dumps({
                    "status": "promoted", "attestation": {"source_commit": "a" * 40}}))
                prelude = f"import shutil, types\nshutil.disk_usage = lambda path: types.SimpleNamespace(free={free})\n"
                environment = {**os.environ, "GITHUB_WORKSPACE": temporary, "RUNNER_TEMP": temporary,
                               "CANDIDATE_VARIANT": variant}
                result = subprocess.run([sys.executable, "-I", "-S", "-c", prelude + code],
                                        cwd=root, env=environment, capture_output=True, text=True, timeout=30)
                receipt = root / "dist/builder-delivery/capacity.json"
                self.assertEqual(result.returncode == 0, expected is not None, result.stderr)
                if expected is None:
                    self.assertFalse(receipt.exists())
                else:
                    value = json.loads(receipt.read_text())
                    self.assertEqual(value["variant"], variant)
                    self.assertEqual(value["required_free_bytes"], expected)
                    self.assertEqual(set(value["observed_free_bytes"].values()), {free})
                self.assertFalse((root / "injected").exists())

    def test_each_large_transfer_checks_variant_capacity_before_download(self):
        for job in (self.main["sign"], self.main["retain"], self.called["qualify"]):
            sequence = steps(job)
            capacity = next(i for i, s in enumerate(sequence) if "policy.capacity_bytes" in runs(s))
            download = next(i for i, s in enumerate(sequence) if "uses: actions/download-artifact@" in s)
            self.assertLess(capacity, download)
            self.assertIn("CANDIDATE_VARIANT: ${{ inputs.variant }}", sequence[capacity])
            self.assertIn("policy_for(os.environ['CANDIDATE_VARIANT'])", runs(sequence[capacity]))
        for workflow in (PRODUCER, MAIN, CALLED):
            self.assertNotIn("docker system prune", workflow)
            self.assertNotIn("docker image prune", workflow)

    def test_producer_build_command_uses_selected_variant_and_exact_source(self):
        step = next(s for s in steps(self.producer) if "'packaging/native-runtime/build.py'" in runs(s))
        code = python_body(step)
        for variant, success in (("cpu", True), ("cuda", True), ("other", False)):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                copy_policy(root)
                (root / "packaging/native-runtime/builders.json").write_text(json.dumps({
                    role: {"image": "sha256:" + "a" * 64} for role in ("bindings", "deb", "rpm")}))
                prelude = ("import subprocess, json\nfrom pathlib import Path\n"
                           "subprocess.check_output = lambda *a, **k: '" + "b" * 40 + "\\n'\n"
                           "subprocess.run = lambda command, **k: Path('command.json').write_text(json.dumps(command))\n")
                environment = {**os.environ, "CANDIDATE_VARIANT": variant, "CANDIDATE_REVISION": "17",
                               "GITHUB_SHA": "b" * 40}
                result = subprocess.run([sys.executable, "-I", "-S", "-c", prelude + code],
                                        cwd=root, env=environment, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode == 0, success, result.stderr)
                if success:
                    command = json.loads((root / "command.json").read_text())
                    for flag, expected in (("--variant", variant), ("--source", "b" * 40),
                                           ("--package-revision", "17"), ("--output", f"dist/native-{variant}"),
                                           ("--cache", f"dist/native-{variant}-cache")):
                        self.assertEqual(command[command.index(flag) + 1], expected)
                    self.assertNotIn("--runtime", command)
                else:
                    self.assertFalse((root / "command.json").exists())

    def test_qualification_passes_selected_variant_and_rejects_mixed_bundle(self):
        step = next(s for s in steps(self.called["qualify"]) if "'packaging/native-runtime/qualify_candidate.py'" in runs(s))
        code = python_body(step)
        cases = (("cpu", "cpu", "cpu", True), ("cuda", "cuda", "cuda", True),
                 ("cpu", None, "cpu", True), ("cuda", None, "cuda", False),
                 ("cpu", "cuda", "cuda", False), ("cuda", "cpu", "cpu", False),
                 ("cuda", "cuda", "cpu", False))
        for selected, declared, suffix, success in cases:
            with self.subTest(selected=selected, declared=declared, suffix=suffix), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                copy_policy(root)
                bundle = root / "dist/signed"
                bundle.mkdir(parents=True)
                candidate = {"version": f"1.5.3-17.{suffix}", "producer": {"source": "a" * 40},
                    "packages": {"rpm": {"path": "rpm/fixture.rpm", "sha256": "b" * 64}},
                    "runtime_manifest_sha256": "c" * 64, "recovery_helper_sha256": "d" * 64}
                if declared is not None:
                    candidate["variant"] = declared
                (bundle / "candidate.json").write_text(json.dumps(candidate))
                fixture = root / "dist/fixture"
                fixture.mkdir()
                for name in ("image.id", "fixture.json", "fixture-packages.txt"):
                    (fixture / name).write_text("fixture\n")
                (root / "dist/evidence").mkdir()
                prelude = ("import subprocess, json\nfrom pathlib import Path\n"
                           "subprocess.run = lambda command, **k: Path('command.json').write_text(json.dumps(command))\n")
                environment = {**os.environ, "CANDIDATE_VARIANT": selected, "PACKAGE_FAMILY": "rpm"}
                result = subprocess.run([sys.executable, "-I", "-S", "-c", prelude + code],
                                        cwd=root, env=environment, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode == 0, success, result.stderr)
                if success:
                    command = json.loads((root / "command.json").read_text())
                    self.assertEqual(command[command.index("--variant") + 1], selected)
                    self.assertEqual(command[command.index("--package-revision") + 1], "17")
                    self.assertEqual(command[command.index("--artifact-sha256") + 1], "b" * 64)
                else:
                    self.assertFalse((root / "command.json").exists())

    def test_variant_is_explicit_at_every_trusted_transition_and_artifact_name(self):
        for workflow in (PRODUCER, MAIN):
            inputs = blocks(workflow.split("    inputs:\n", 1)[1].split("\npermissions:", 1)[0], 6)
            self.assertIn("        options: [cpu, cuda]\n", inputs["variant"])
            self.assertIn("        default: cpu\n", inputs["variant"])
        for job, operation in ((self.main["validate"], "validate_candidate.py"),
                               (self.main["sign"], "finalize_candidate.py sign"),
                               (self.called["qualify"], "finalize_candidate.py verify"),
                               (self.main["retain"], "finalize_candidate.py complete")):
            step = next(s for s in steps(job) if operation in runs(s))
            self.assertIn("CANDIDATE_VARIANT: ${{ inputs.variant }}", step)
            self.assertIn('--variant "$CANDIDATE_VARIANT"', runs(step))
        for workflow in (PRODUCER, MAIN, CALLED):
            for job in jobs(workflow).values():
                for step in steps(job):
                    if "uses: actions/upload-artifact@" in step:
                        self.assertIn("name: native-${{ inputs.variant }}-", step)
        upload = next(s for s in steps(self.producer) if "uses: actions/upload-artifact@" in s)
        self.assertIn("dist/native-${{ inputs.variant }}/deb/*.deb", upload)
        self.assertIn("dist/native-${{ inputs.variant }}/rpm/build/RPMS/x86_64/*.rpm", upload)

    def test_manual_entry_and_reusable_entry_are_separate(self):
        self.assertEqual(set(blocks(MAIN.split("\non:\n", 1)[1].split("\npermissions:", 1)[0], 2)),
                         {"workflow_dispatch"})
        self.assertEqual(set(blocks(CALLED.split("\non:\n", 1)[1].split("\npermissions:", 1)[0], 2)),
                         {"workflow_call"})
        for job in (self.producer, self.main["validate"], self.main["sign"], self.called["qualify"]):
            self.assertIn("if: github.repository == 'Hkshoonya/nvidia-broadcast-linux' && github.ref == 'refs/heads/main'", job)

    def test_dispatch_requires_exact_producer_tuple_and_revision(self):
        declaration = MAIN.split("    inputs:\n", 1)[1].split("\npermissions:", 1)[0]
        inputs = blocks(declaration, 6)
        self.assertEqual(set(inputs), {"producer_run", "producer_attempt", "source_commit", "artifact_id",
                                      "artifact_digest", "package_revision", "variant"})
        self.assertIn("        type: choice\n", inputs["variant"])
        self.assertIn("        options: [cpu, cuda]\n", inputs["variant"])
        self.assertIn("        required: true\n", inputs["variant"])
        self.assertIn("        default: cpu\n", inputs["variant"])
        for name, body in inputs.items():
            if name == "variant":
                continue
            with self.subTest(input=name):
                self.assertIn("        type: string\n", body)
                self.assertIn("        required: true\n", body)
                self.assertNotIn("default:", body)

    def test_reusable_qualification_has_no_secret_or_environment_contract(self):
        declaration = CALLED.split("    inputs:\n", 1)[1].split("    outputs:\n", 1)[0]
        inputs = blocks(declaration, 6)
        self.assertEqual(set(inputs), {"family", "signed_artifact_id", "candidate_manifest", "variant"})
        for name, body in inputs.items():
            self.assertIn("        type: string\n", body)
            self.assertIn("        required: false\n" if name == "variant" else "        required: true\n", body)
        self.assertIn("        default: cpu\n", inputs["variant"])
        self.assertNotIn("secrets:", CALLED)
        self.assertNotIn("secrets.", CALLED)
        self.assertNotIn("environment:", CALLED)
        self.assertNotIn("GH_TOKEN", CALLED)

    def test_workflow_permissions_are_read_only_without_release_or_oidc_writes(self):
        for workflow in (PRODUCER, MAIN, CALLED):
            with self.subTest(workflow=workflow.splitlines()[0]):
                self.assertIn("permissions:\n  contents: read\n", workflow)
                self.assertNotRegex(workflow, r"(?m)^\s+(?:[a-z-]+):\s*write\s*$")
                self.assertNotIn("write-all", workflow)
                self.assertNotIn("id-token:", workflow)
                self.assertNotIn("gh release", workflow)
                self.assertNotIn("git push", workflow)
                self.assertNotIn("softprops/action-gh-release", workflow)

    def test_remote_actions_are_pinned_and_checkout_uses_finalizer_revision(self):
        for workflow in (PRODUCER, MAIN, CALLED):
            for action in re.findall(r"(?m)^\s+(?:- )?uses: (\S+)", workflow):
                if action.startswith("./"):
                    self.assertEqual(action, "./.github/workflows/native-cpu-qualify.yml")
                else:
                    self.assertRegex(action, r"^[A-Za-z0-9_-]+/[A-Za-z0-9_-]+@[0-9a-f]{40}$")
            for job in jobs(workflow).values():
                for step in steps(job):
                    if "uses: actions/checkout@" in step:
                        self.assertIn("ref: ${{ github.sha }}", step)
                        self.assertIn("persist-credentials: false", step)
                        self.assertNotIn("inputs.source_commit", step)

    def test_signing_is_protected_and_only_one_step_receives_credentials(self):
        self.assertEqual(MAIN.count("environment: rpm-signing"), 1)
        self.assertIn("environment: rpm-signing", self.main["sign"])
        secret_steps = [(job_name, step) for job_name, job in self.main.items()
                        for step in steps(job) if "secrets." in step]
        self.assertEqual(len(secret_steps), 1)
        job_name, step = secret_steps[0]
        self.assertEqual(job_name, "sign")
        self.assertEqual(set(re.findall(r"secrets\.([A-Z_]+)", step)),
                         {"RPM_SIGNING_PRIVATE_KEY", "RPM_SIGNING_PASSPHRASE"})
        command = runs(step)
        self.assertIn("packaging/native-runtime/finalize_candidate.py sign", command)
        self.assertIn('--expected-receipt "$VALIDATION_RECEIPT"', command)
        for forbidden in ("docker", "sudo", "apt-get", "qualify_candidate", "pip", "import dist", "dist/signed/"):
            self.assertNotIn(forbidden, command)
        sign_commands = "\n".join(runs(s) for s in steps(self.main["sign"]))
        self.assertNotRegex(sign_commands, r"(?:dpkg\s+-i|rpm\s+(?:-i|-U|--install|--upgrade)|docker\s+run)")
        self.assertNotIn("qualify_candidate.py", sign_commands)

    def test_inputs_enter_commands_only_through_step_environment(self):
        for workflow in (PRODUCER, MAIN, CALLED):
            for job in jobs(workflow).values():
                for step in steps(job):
                    self.assertNotIn("${{", runs(step), "interpolated expression inside executable script")
        validation = next(s for s in steps(self.main["validate"]) if "validate_candidate.py" in runs(s))
        for input_name, variable in (("producer_run", "PRODUCER_RUN"), ("producer_attempt", "PRODUCER_ATTEMPT"),
                                    ("source_commit", "SOURCE_COMMIT"), ("artifact_id", "ARTIFACT_ID"),
                                    ("artifact_digest", "ARTIFACT_DIGEST"), ("package_revision", "PACKAGE_REVISION"),
                                    ("variant", "CANDIDATE_VARIANT")):
            self.assertIn(variable + ": ${{ inputs." + input_name + " }}", validation)
            self.assertIn('"$' + variable + '"', runs(validation))
        for job_name, job in self.main.items():
            token_steps = [s for s in steps(job) if "GH_TOKEN:" in s]
            if job_name == "validate":
                self.assertEqual(len(token_steps), 2)
            else:
                self.assertEqual(token_steps, [])

    def test_dependency_graph_cannot_retain_partial_qualification(self):
        self.assertIn("    needs: validate\n", self.main["sign"])
        for family in ("deb", "rpm"):
            job = self.main["qualify-" + family]
            self.assertIn("    needs: sign\n", job)
            self.assertIn("    uses: ./.github/workflows/native-cpu-qualify.yml\n", job)
            self.assertIn("      family: " + family + "\n", job)
            self.assertIn("      variant: ${{ inputs.variant }}\n", job)
            self.assertIn("signed_artifact_id: ${{ needs.sign.outputs.artifact_id }}", job)
            self.assertIn("candidate_manifest: ${{ needs.sign.outputs.manifest }}", job)
            self.assertNotIn("secrets:", job)
        self.assertIn("    needs: [sign, qualify-deb, qualify-rpm]\n", self.main["retain"])
        for workflow in (MAIN, CALLED):
            self.assertNotIn("always()", workflow)
            self.assertNotIn("continue-on-error", workflow)

    def test_transfers_use_immutable_ids_to_survive_partial_reruns(self):
        for job_name in ("validate", "sign"):
            job = self.main[job_name]
            self.assertIn("artifact_id: ${{ steps.transfer.outputs.artifact-id }}", job)
            transfer = next(s for s in steps(job) if "uses: actions/upload-artifact@" in s)
            self.assertIn("id: transfer", transfer)
        self.assertIn("artifact_id: ${{ steps.transfer.outputs.artifact-id }}", self.called["qualify"])
        self.assertIn("value: ${{ jobs.qualify.outputs.artifact_id }}", CALLED)
        for workflow in (MAIN, CALLED):
            for job in jobs(workflow).values():
                for step in steps(job):
                    if "uses: actions/download-artifact@" in step:
                        self.assertIn("artifact-ids: ${{", step)
                        self.assertIn("merge-multiple: true", step)
                        self.assertNotRegex(step, r"(?m)^\s+(?:name|pattern|run-id|github-token):")
        downloads = "\n".join(s for s in steps(self.main["retain"]) if "actions/download-artifact@" in s)
        for producer in ("sign", "qualify-deb", "qualify-rpm"):
            self.assertIn("${{ needs." + producer + ".outputs.artifact_id }}", downloads)

    def test_signed_identity_is_verified_before_fixture_build_and_execution(self):
        sequence = steps(self.called["qualify"])
        verify = next(i for i, s in enumerate(sequence) if "finalize_candidate.py verify" in runs(s))
        build = next(i for i, s in enumerate(sequence) if "fixtures/build_fixture.py" in runs(s))
        execute = next(i for i, s in enumerate(sequence) if "qualify_candidate.py" in runs(s))
        self.assertLess(verify, build)
        self.assertLess(build, execute)
        self.assertIn("CANDIDATE_MANIFEST: ${{ inputs.candidate_manifest }}", sequence[verify])
        self.assertIn('--expected-manifest "$CANDIDATE_MANIFEST"', runs(sequence[verify]))
        self.assertIn("'--evidence-output', 'dist/evidence'", runs(sequence[execute]))

    def test_only_completed_candidate_gets_long_retention(self):
        for workflow in (MAIN, CALLED):
            for job_name, job in jobs(workflow).items():
                for step in steps(job):
                    if "uses: actions/upload-artifact@" in step:
                        if "if: failure()" in step:
                            self.assertIn("native-${{ inputs.variant }}-unqualified-diagnostics-", step)
                            self.assertIn("if-no-files-found: warn", step)
                            self.assertNotIn("dist/execution/\n", step)
                            self.assertNotIn("/probes", step)
                            self.assertNotIn("/inputs", step)
                        else:
                            self.assertIn("if-no-files-found: error", step)
                        self.assertNotIn("overwrite:", step)
                        if job_name == "retain":
                            self.assertIn("retention-days: 30", step)
                            self.assertIn("path: dist/qualified/", step)
                        else:
                            self.assertIn("retention-days: 1", step)
        sequence = steps(self.main["retain"])
        complete = next(i for i, s in enumerate(sequence) if "finalize_candidate.py complete" in runs(s))
        upload = next(i for i, s in enumerate(sequence) if "uses: actions/upload-artifact@" in s)
        self.assertLess(complete, upload)
        self.assertIn("CANDIDATE_MANIFEST: ${{ needs.sign.outputs.manifest }}", sequence[complete])


if __name__ == "__main__":
    unittest.main()
