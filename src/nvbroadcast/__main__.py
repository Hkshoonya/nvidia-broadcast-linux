# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/doczeus)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""Entry point for NVIDIA Broadcast."""

import sys


def main():
    from nvbroadcast.app import NVBroadcastApp

    app = NVBroadcastApp()
    return app.run(sys.argv)


if __name__ == "__main__":
    sys.exit(main())
