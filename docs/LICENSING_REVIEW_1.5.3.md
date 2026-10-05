# Licensing review packet for v1.5.3

Prepared 5 October 2026 for
[#100](https://github.com/Hkshoonya/nvidia-broadcast-linux/issues/100).
This records the maintainer's decision and source/packaging evidence. It is
not an independent legal opinion from the release tooling.

## Maintainer decision — 5 October 2026

The maintainer confirmed that a legal friend completed the review and directed
work to proceed. No revised terms or exclusions were supplied. The candidate
therefore retains the existing project license grant, attribution requirements,
creator/contributor notices, and separate macOS Camera Extension license.
The reviewer was described as a legal friend; no written opinion or itemized
codec/model/NVIDIA distribution scope was provided in this conversation.

The grant already permits GPL version 3 or any later version. The candidate
makes the existing `-or-later` declarations consistent, includes an unmodified
copy of the complete GPLv3 text after the existing LICENSE content, and keeps
the attribution terms intact. The complete text was retrieved from the
[Free Software Foundation](https://www.gnu.org/licenses/gpl-3.0.txt), SHA-256
`3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986`.
These changes supply missing distributed text and accurate references; they
do not remove attribution requirements or relicense third-party components.

The technical artifact audit must still inventory the exact redistributed
components and their notices. Record verified component scope precisely;
do not expand the maintainer's confirmation into an independently verified
legal opinion covering every binary or model.

## Confirmed third-party notice correction

`LICENSE` previously described RobustVideoMatting as MIT. The app's RVM model
URLs in `src/nvbroadcast/video/effects.py` point to the upstream `v1.0.0`
release. That source tag resolves to
`17d1774b032fd503bfe53c57d295db719f9e3da1`, whose
[LICENSE is the GNU GPLv3 text](https://github.com/PeterL1n/RobustVideoMatting/blob/17d1774b032fd503bfe53c57d295db719f9e3da1/LICENSE).
The upstream README also records its GPLv3 source re-release.

The candidate corrects the **source** license label. That factual correction
does not establish the separate scope or redistribution terms of every ONNX
model asset. Review the exact model release, hashes, notices, and any separate
weight terms before recording model-distribution clearance. The runtime
downloads weights; release recipes and cache behavior must also be checked to
determine which artifacts actually contain them.

## Project arrangement to review

The retained `LICENSE` contains a GPLv3-or-later notice and project-specific
requirements to retain the complete file, source copyright headers, UI creator
credit, author metadata, and original project URL. The candidate now also includes the complete standard GPLv3 text. `NOTICE` preserves the
canonical upstream, original creator, accepted external contributors, and
NVIDIA non-affiliation, and references the current attribution requirements.

[GPLv3 section 7](https://www.gnu.org/licenses/gpl.en.html#section7) describes
categories of additional terms, including some attribution/legal-notice
conditions, and treatment of further restrictions. Applying that section to
this project's exact mandatory wording requires review; this packet does not
conclude that every current condition is permitted or prohibited.

The maintainer chose to proceed with the existing arrangement. Creator and
accepted-contributor credits remain in `NOTICE`, and mandatory attribution
wording remains in `LICENSE`. Python, source-header, Debian, RPM, Snap,
AppStream, website and Flatpak references describe the existing GPLv3-or-later
grant; the Nix draft already uses `gpl3Plus`. GitHub's heuristic detection may
continue to report `NOASSERTION` for a file containing additional terms. Do not
remove those terms merely to make automatic detection succeed.

## Recording and runtime inventory to review

| Delivery path | Technical evidence | Licensing questions still requiring clearance |
| --- | --- | --- |
| Snap recording part | Stages Core24 GStreamer recording plugins, OpenH264 and VisualOn AAC; explicitly retains the selected package copyright files and VisualOn notice | Exact staged package versions and corresponding source; code and codec/patent distribution conditions; whether retained notices are complete |
| DEB | Uses native GStreamer packages; the tested recording path used x264 and VisualOn AAC | Distribution of the application and any bundled materials; distinguish host-installed packages from files redistributed inside our artifact |
| RPM | Uses Fedora packages; the tested path used OpenH264 and FDK AAC | Exact provider/repository and source/notice obligations, including separately supplied codec packages |
| Flatpak CPU | GNOME runtime plus a pinned Python graph; tested x264/avenc_aac recording | Which codecs come from the runtime versus app payload, notices and source obligations, and Flathub's independent source-build policy |
| CUDA Snap/runtime prototype | NVIDIA CUDA/cuDNN components, CuPy, ONNX Runtime GPU; optional TensorRT is a separate input | Exact NVIDIA redistribution agreements and version scope, complete notices, and compatibility with the reviewed application arrangement |
| Models | RVM ONNX release URLs and hash-verified faster-whisper snapshots | Review weights separately from repository/source-code license labels and identify whether each package bundles or downloads them |

The native prototype retains standalone-Python upstream license records and
Python distribution metadata. That inventory is useful review material, not
proof that every binary, native dependency, model, or target is cleared.

## Remaining technical verification

Inspect the exact source archive, wheel, native packages, Snap and Flatpak for
the complete project LICENSE, NOTICE, accepted contributor credits and relevant
third-party notices. Run the package/store lint and record any rejected license
identifier or unsupported additional-term handling. Preserve the maintainer's
review confirmation above without inventing a reviewer identity, written
approval, or unreported scope. #100 remains open until the artifact and metadata
checks are recorded; further legal input is needed only if that work exposes a
specific unresolved distribution condition.
