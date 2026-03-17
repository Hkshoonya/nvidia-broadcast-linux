# Contributing to NVIDIA Broadcast for Linux

Thank you for your interest in contributing! This project was built for the Linux community and contributions are welcome.

## Ground Rules

1. **All PRs require review** — No direct pushes to `main`. Every change goes through a pull request and needs approval from a maintainer.

2. **Keep attribution intact** — The `doczeus` copyright headers, UI credits, and LICENSE file must not be removed or modified. This is a legal requirement of the GPL-3.0 license.

3. **Be respectful** — We're all here because we love Linux and want better tools. Constructive feedback only.

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

### Code Style

- Python 3.11+ with type hints where helpful
- Follow existing file structure and naming conventions
- Copyright header on new `.py` files:
  ```python
  # NVIDIA Broadcast for Linux
  # Copyright (c) 2026 doczeus (https://github.com/doczeus)
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
