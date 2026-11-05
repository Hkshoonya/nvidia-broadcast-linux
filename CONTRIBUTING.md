# Contributing to NVIDIA Broadcast for Linux

Thank you for your interest in contributing! This project was built for the Linux community and contributions are welcome.

## Ground Rules

1. **All PRs require review** — No direct pushes to `main`. Every change goes through a pull request and needs approval from a maintainer.

2. **Keep attribution intact** — The `doczeus` copyright headers, UI credits, and LICENSE file must not be removed or modified. These requirements are part of this project's license terms; see [LICENSE](LICENSE).

3. **Be respectful** — We're all here because we love Linux and want better tools. Constructive feedback only.

4. **Contributor credits follow accepted work** — Accepted external human contributions must register their author in `src/nvbroadcast/contributors.py` and describe the accepted work in `CONTRIBUTORS.md` before merge. The app shows **Core Maintainers**, **Major Feature Contributors**, and **Contributors**, with evidence-backed scope credit. Verified fixes are eligible for recognition; merely submitting a pull request or passing CI does not establish acceptance. Sponsors remain separate under **Project Sponsors**.

## How to Contribute

### Reporting Bugs

- Use the [Bug Report](https://github.com/Hkshoonya/nvidia-broadcast-linux/issues/new?template=bug_report.md) template
- Include your system info, GPU model, and terminal output
- Screenshots/video help a lot for visual bugs

### Suggesting Features

- Use the [Feature Request](https://github.com/Hkshoonya/nvidia-broadcast-linux/issues/new?template=feature_request.md) template
- Explain why the feature would be useful

### Submitting Code

1. **Fork** the repository
2. **Create a feature branch**: `git checkout -b feature/my-feature`
3. **Make your changes** following the code style below
4. **Test locally**: make sure the app runs, effects work, no regressions
5. **Commit** with clear messages
6. **Push** to your fork and open a **Pull Request**

### Review and Merge Priorities

Prioritize verified reliability, camera/audio, performance, security and
installation problems over cosmetic work. Each pull request needs a
reproducible problem or a meaningful benefit, a focused change and appropriate
verification. A small diff that fixes an important verified bug remains
eligible; contribution depth and merge priority are separate judgments.

Unrequested cosmetic churn, duplicates and standalone self-credit requests
without accepted work are not sufficient reasons to merge. Legitimate
attribution corrections are still maintained. Passing CI or adding a credit
entry does not establish acceptance or replace maintainer review.

### Contributor Credit

For accepted external human work, add one `Contributor` entry with your public
display name and GitHub login to `src/nvbroadcast/contributors.py`, unless you
are already listed, and describe the accepted work in `CONTRIBUTORS.md`.
Include a concise scope, contribution areas and links to the accepted pull
requests, commits or validated finding. Keep the README credit summary in sync.
Pull-request checks verify the cumulative records before merge.

Maintainers review the evidence and display group. Substantial features,
architecture, integration work, and technically significant verified fixes can
receive **Major Feature Contributors** recognition. Focused fixes, compatibility
work, documentation and validated diagnoses receive concise **Contributors**
scope credit. **Core Maintainers** describes ongoing project responsibility.
No grouping is calculated from lines changed, pull-request totals, account age
or sponsorship, and a display group grants no repository or review privileges.

Technical depth, source acceptance and release delivery are separate facts.
Unmerged proposals are not credited as delivered work; a validated finding can
be credited even when its proposed implementation was superseded. Keep release
and platform limits explicit. Maintainer and automated dependency-bot changes
retain their existing credit-gate exemptions. All accepted credits remain
cumulative in future releases.

### Code Style

- Python 3.11+ with type hints where helpful
- Follow existing file structure and naming conventions
- Copyright header on new `.py` files:
  ```python
  # NVIDIA Broadcast for Linux
  # Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
  # Licensed under GPL-3.0 - see LICENSE file
  # Original author: doczeus | AI Powered
  #
  ```
- Keep it simple — don't over-engineer

### Areas Where Help Is Needed

- Better segmentation models
- Eye contact correction
- Virtual lighting
- System tray indicator
- Flatpak / Snap packaging
- Multi-camera support
- Performance optimizations
- Documentation and translations

## Code of Conduct

- Be kind and constructive
- No harassment, trolling, or personal attacks
- Focus on the code and the mission: making Linux broadcast-ready

---

**Created by [Doczeus](https://github.com/Hkshoonya)** — contributions are appreciated, attribution is required.
