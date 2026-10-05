# Licensing review packet for v1.5.3

Prepared 5 October 2026 for
[#100](https://github.com/Hkshoonya/nvidia-broadcast-linux/issues/100).
This records source and packaging evidence and the decisions needed from a
qualified license reviewer and the maintainer. It is not distribution clearance
and does not change the project's license or attribution requirements.

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

The current `LICENSE` contains a GPLv3-or-later notice and project-specific
requirements to retain the complete file, source copyright headers, UI creator
credit, author metadata, and original project URL. It links to the standard
GPL text rather than including that complete text. `NOTICE` preserves the
canonical upstream, original creator, accepted external contributors, and
NVIDIA non-affiliation, and references the current attribution requirements.

[GPLv3 section 7](https://www.gnu.org/licenses/gpl.en.html#section7) describes
categories of additional terms, including some attribution/legal-notice
conditions, and treatment of further restrictions. Applying that section to
this project's exact mandatory wording requires review; this packet does not
conclude that every current condition is permitted or prohibited.

The concrete proposal to assess is complete standard GPL-3.0-or-later text in
`LICENSE`, factual copyright and contributor attribution in `NOTICE`, and any
reviewed additional terms stated separately with accurate scope. The reviewer
must decide which existing conditions can be retained, whether consent from
other rights holders is needed, and which SPDX expression describes the final
arrangement. The maintainer must approve the resulting wording before it is
applied. Factual creator and accepted-contributor credits must be preserved.

After that decision, synchronize Python, source headers, Debian, RPM, Snap,
AppStream, website, Flatpak and the Nix draft, include the full applicable
texts, and verify contents in built artifacts. Keep the separately licensed
macOS Camera Extension explicitly scoped.

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

## Required decision record

Record the reviewer, date, exact project wording and dependency/artifact
versions assessed, approved license/notice arrangement, any distribution
conditions, and remaining exclusions. Then attach the approved result to #100
and implement one consistent metadata/text change. Until that record exists,
the v1.5.3 release and any public Flatpak distribution remain unqualified on
this licensing gate.
