"""Check launcher quoting and CUDA library isolation with a real subprocess."""
from pathlib import Path
import json
import os
import subprocess

from scripts.install_hush_fork import atomic_write, worker_launcher


def test_worker_launcher_preserves_paths_and_isolates_environment(tmp_path):
    root = tmp_path / 'spaces $dollar "quotes"'
    root.mkdir()
    binary = root / 'worker'
    atomic_write(binary, '''#!/usr/bin/env python3
import json,os,sys
print(json.dumps([os.environ['NVAFX_MODEL'], os.environ['LD_LIBRARY_PATH'],
                  os.environ['CUDA_VISIBLE_DEVICES'], sys.argv[1:]]))
''')
    sdk = root / 'sdk'
    model = root / 'model.trtpkg'
    launcher = root / 'launcher'
    atomic_write(launcher, worker_launcher(binary, sdk, model, 'GPU-test'))
    environment = {**os.environ, 'LD_LIBRARY_PATH': '/existing/library'}
    result = subprocess.run([str(launcher), 'a b', '$(touch unexpected)'],
                            env=environment, check=True, capture_output=True, text=True)
    values = json.loads(result.stdout)
    assert values == [str(model), ':'.join(str(sdk / p) for p in (
        'nvafx/lib', 'external/cuda/lib', 'features/denoiser/lib')) + ':/existing/library',
        'GPU-test', ['a b', '$(touch unexpected)']]
    assert environment['LD_LIBRARY_PATH'] == '/existing/library'
    assert not (Path.cwd() / 'unexpected').exists()


def test_stop_removes_only_the_old_hush_virtual_microphone():
    from unittest import mock
    from scripts import install_hush_fork as installer
    modules = (
        "10\tmodule-null-sink\tsink_name=HUSH channel_map=mono\tn/a\n"
        "11\tmodule-null-sink\tsink_name=nvbroadcast_sink\tn/a\n"
        "12\tmodule-null-sink\tsink_name=HUSH_OTHER\tn/a\n"
    )
    with mock.patch.object(installer.shutil, 'which',
                           side_effect=lambda name: '/usr/bin/pactl' if name == 'pactl' else None), \
            mock.patch.object(installer.subprocess, 'run', return_value=
                              subprocess.CompletedProcess([], 1, '', '')), \
            mock.patch.object(installer, 'output', return_value=modules), \
            mock.patch.object(installer, 'run') as run, \
            mock.patch.object(installer.time, 'sleep'):
        installer.stop_hush()
    run.assert_called_once_with('pactl', 'unload-module', '10')
