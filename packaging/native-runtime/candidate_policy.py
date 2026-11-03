"""Trusted, explicitly selected bounds for hosted native development candidates.

CPU limits preserve the original qualification contract. CUDA limits accommodate
the measured revision 6 pair (4.13 GB archives, 5.01 GB regular payload) while
keeping transport, expansion and disk use finite. The 40 GiB CUDA disk gate is a
conservative allocation budget for builders, cache, staging and private copies;
it is not a claim about a measured hosted peak. No policy is selected from an
artifact's contents.
"""
from typing import NamedTuple

GIB = 1024**3
MIB = 1024**2


class CandidatePolicy(NamedTuple):
    variant: str
    package_max_bytes: int
    transport_max_bytes: int
    payload_max_bytes: int
    tar_max_bytes: int
    capacity_bytes: int


def policy_for(variant):
    if variant == "cpu":
        return CandidatePolicy("cpu", GIB, 2*GIB, 4*GIB, 5*GIB, 12*GIB)
    if variant == "cuda":
        # Per-package release ceiling is strict; transport holds both packages,
        # at most 32 MiB provenance, and bounded ZIP framing overhead.
        return CandidatePolicy("cuda", 2*GIB-1, 4*GIB+64*MIB, 6*GIB, 7*GIB, 40*GIB)
    raise ValueError("candidate variant must be cpu or cuda")


def check_variant(record, variant):
    policy_for(variant)
    actual = record.get("variant", "cpu")
    if actual != variant:
        raise ValueError("candidate variant differs from explicit selection")


def validate_runtime_probe(probe, variant):
    policy_for(variant)
    if not isinstance(probe, dict) or probe.get("status") != "pass":
        raise ValueError("runtime probe did not pass")
    check_variant(probe, variant)
    if variant == "cpu":
        # Historical CPU receipts predate explicit provider fields.
        return
    cpu, cuda = probe.get("cpu_execution"), probe.get("cuda_execution")
    if not (probe.get("cuda_unavailable_expected") is True
            and isinstance(cpu, dict) and cpu.get("provider") == "cpu"
            and cpu.get("success") is True
            and cpu.get("executed_providers") == ["CPUExecutionProvider"]
            and isinstance(cuda, dict) and cuda.get("provider") == "cuda"
            and cuda.get("success") is False
            and isinstance(cuda.get("executed_providers"), list)
            and "CUDAExecutionProvider" not in cuda["executed_providers"]
            and probe.get("cupy_execution") is None):
        raise ValueError("CUDA no-device probe lacks CPU execution or expected CUDA rejection")
