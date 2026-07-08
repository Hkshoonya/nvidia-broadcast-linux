# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""Entry point for NVIDIA Broadcast."""

import sys


def main():
    from nvbroadcast.core.startup_trace import mark

    mark("python entry")
    from nvbroadcast.app import NVBroadcastApp

    mark("modules imported")
    app = NVBroadcastApp()
    mark("app constructed")
    return app.run(sys.argv)


if __name__ == "__main__":
    sys.exit(main())
