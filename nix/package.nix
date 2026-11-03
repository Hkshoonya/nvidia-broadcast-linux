{
  lib,
  stdenv,
  autoPatchelfHook,
  fetchFromGitHub,
  fetchurl,
  findutils,
  python313Packages,
  runCommand,
  wrapGAppsHook4,
  gobject-introspection,
  cairo,
  coreutils,
  gawk,
  gdk-pixbuf,
  gitMinimal,
  glib,
  graphene,
  gst_all_1,
  gtk4,
  libadwaita,
  libglvnd,
  libxcb,
  mailcap,
  makeShellWrapper,
  onnx,
  pango,
  pipewire,
  psmisc,
  protobuf_33,
  pulseaudio,
  v4l-utils,
  writableTmpDirAsHomeHook,
  writeShellScript,
  zlib,
}:

let
  qualification = lib.importJSON ./security-inputs.json;
  securityInputs = qualification.packages;
  nativePlatforms = {
    build = stdenv.buildPlatform;
    host = stdenv.hostPlatform;
    target = stdenv.targetPlatform;
  };
  nativeOnly = lib.all (platform:
    platform.system == "x86_64-linux" && platform.config == stdenv.buildPlatform.config
  ) (builtins.attrValues nativePlatforms);
  spliceNames = [
    "pythonOnBuildForBuild"
    "pythonOnBuildForHost"
    "pythonOnBuildForTarget"
    "pythonOnHostForHost"
    "pythonOnTargetForTarget"
  ];
  python = (python313Packages.python.override {
    self = python;
    sourceVersion = { major = "3"; minor = "13"; patch = "18"; suffix = ""; };
    inherit (qualification.python) hash;
    # Pinned Nixpkgs propagates scalar hash but drops attrset sourceVersion
    # when deriving default splices. In this native-only package all aliases
    # must select this same interpreter, including its exact source identity.
    __splices = lib.genAttrs spliceNames (_: python);
  }).overrideAttrs (old: {
    patches =
      let isMimetypesPatch = patch: builtins.baseNameOf (toString patch) == "mimetypes.patch";
      in assert lib.assertMsg (builtins.length (builtins.filter isMimetypesPatch old.patches) == 1)
        "Expected exactly one Nixpkgs mimetypes patch to port";
      map (patch: if isMimetypesPatch patch then ./mimetypes-3.13.18.patch else patch) old.patches;
  });
  pinnedSource = name: fetchurl {
    inherit (securityInputs.${name}) url hash;
  };

  # Bootstrap this pure-Python wheel without changing the interpreter ABI.
  # The same selected version is used by build backends and runtime consumers.
  secureSetuptools = python.pkgs.buildPythonPackage {
    pname = "setuptools";
    inherit (securityInputs.setuptools) version;
    format = "wheel";
    src = pinnedSource "setuptools";
    pythonImportsCheck = [ "setuptools" ];
    meta.license = lib.licenses.mit;
  };

  pythonPackages = python.pkgs.overrideScope (_final: previous: {
    setuptools = secureSetuptools;
    pillow = previous.pillow.overridePythonAttrs (old: {
      inherit (securityInputs.pillow) version;
      src = pinnedSource "pillow";
      meta = old.meta // {
        changelog = "https://pillow.readthedocs.io/en/stable/releasenotes/13.0.0.html";
      };
    });
    certifi = previous.certifi.overridePythonAttrs (_old: {
      inherit (securityInputs.certifi) version;
      src = pinnedSource "certifi";
      # Keep upstream's fresh bundle and default lookup semantics. External
      # trust configuration is handled explicitly by the application wrapper.
      patches = [ ];
      postPatch = "";
    });
    filelock = previous.filelock.overridePythonAttrs (_old: {
      inherit (securityInputs.filelock) version;
      src = pinnedSource "filelock";
    });
    urllib3 = previous.urllib3.overridePythonAttrs (old: {
      inherit (securityInputs.urllib3) version;
      src = pinnedSource "urllib3";
      meta = old.meta // {
        changelog = "https://github.com/urllib3/urllib3/blob/2.8.0/CHANGES.rst";
      };
    });
  });

  pythonSourceIdentities = lib.mapAttrs (_name: interpreter: {
    version = interpreter.version;
    urls = interpreter.src.urls;
    hash = builtins.convertHash {
      hash = interpreter.src.outputHash;
      hashAlgo = "sha256";
      toHashFormat = "sri";
    };
    source = toString interpreter.src;
  }) ({ selected = pythonPackages.python; } //
    lib.genAttrs spliceNames (name: pythonPackages.python.${name}));
  pythonSourcesMatch = lib.all (identity:
    identity.version == qualification.python.version &&
    identity.urls == [ qualification.python.url ] &&
    identity.hash == qualification.python.hash
  ) (builtins.attrValues pythonSourceIdentities);

  av = pythonPackages.buildPythonPackage {
    pname = "av";
    version = "16.0.1";

    format = "wheel";

    src = fetchurl {
      url = "https://files.pythonhosted.org/packages/35/fd/867ba4cc3ab504442dc89b0c117e6a994fc62782eb634c8f31304586f93e/av-16.0.1-cp313-cp313-manylinux_2_28_x86_64.whl";
      hash = "sha256-h2QVRwpi5KNVDMONsvwAlMJeZO6jTXKTt0VBJdWVgZA=";
    };

    nativeBuildInputs = [ autoPatchelfHook ];
    buildInputs = [
      stdenv.cc.cc.lib
      zlib
    ];

    pythonImportsCheck = [ "av" ];

    meta = {
      description = "Pythonic bindings for FFmpeg libraries";
      homepage = "https://github.com/PyAV-Org/PyAV";
      license = lib.licenses.bsd3;
      sourceProvenance = with lib.sourceTypes; [ binaryNativeCode ];
      platforms = [ "x86_64-linux" ];
    };
  };

  audiolab = pythonPackages.buildPythonPackage {
    pname = "audiolab";
    version = "0.5.1";
    format = "wheel";

    src = fetchurl {
      url = "https://files.pythonhosted.org/packages/4e/4f/e80942de3c6f4338334473fe9d5cf19a07eee967306524fc9708b2336bb2/audiolab-0.5.1-py3-none-any.whl";
      hash = "sha256-ieBUkIOqeEI/n4yhaWNZX3kjtRUR/3fTLutrSOw4hjM=";
    };

    dependencies = with pythonPackages; [
      av
      click
      humanize
      jinja2
      requests
      smart-open
      soundfile
    ];

    pythonImportsCheck = [ "audiolab" ];

    meta = {
      description = "Python library for audio processing built on top of SoundFile and PyAV";
      homepage = "https://github.com/pengzhendong/audiolab";
      license = lib.licenses.asl20;
      sourceProvenance = with lib.sourceTypes; [ binaryNativeCode ];
      platforms = [ "x86_64-linux" ];
    };
  };

  mediapipe = pythonPackages.buildPythonPackage {
    pname = "mediapipe";
    version = "1.0.1";
    format = "wheel";

    src = fetchurl {
      url = "https://files.pythonhosted.org/packages/2a/58/bdd5bada89d7a132375df05e962bf702c148b47043dca98d820d9395152b/mediapipe-1.0.1-py3-none-manylinux_2_28_x86_64.whl";
      hash = "sha256-EhUiJRr8PBNeS3sMNB3V4FCtHsh2MRJ0hPPDia44UEQ=";
    };

    nativeBuildInputs = [ autoPatchelfHook ];

    buildInputs = [
      libglvnd
      stdenv.cc.cc.lib
    ];

    dependencies = with pythonPackages; [
      absl-py
      certifi
      flatbuffers
      matplotlib
      numpy
      opencv-contrib-python
      sounddevice
    ];

    pythonImportsCheck = [ "mediapipe" ];

    meta = {
      description = "Cross-platform framework for building multimodal applied machine learning pipelines";
      homepage = "https://github.com/google-ai-edge/mediapipe";
      license = lib.licenses.asl20;
      sourceProvenance = with lib.sourceTypes; [ binaryNativeCode ];
      platforms = [ "x86_64-linux" ];
    };
  };

  pyrnnoise = pythonPackages.buildPythonPackage {
    pname = "pyrnnoise";
    version = "0.4.3";
    format = "wheel";

    src = fetchurl {
      url = "https://files.pythonhosted.org/packages/04/51/993a25a8b5220e23e0a31ff98747b8fce4685336e0fc4e8e156feab5c4f1/pyrnnoise-0.4.3-py3-none-manylinux1_x86_64.whl";
      hash = "sha256-GwlHd+c3l8XdZHeCkCxpHruaPEVsh450JZf1tVU1o9s=";
    };

    nativeBuildInputs = [ autoPatchelfHook ];
    buildInputs = [ stdenv.cc.cc.lib ];

    dependencies = with pythonPackages; [
      audiolab
      click
      matplotlib
      numpy
      tqdm
    ];

    pythonImportsCheck = [ "pyrnnoise" ];

    meta = {
      description = "Python bindings for RNNoise";
      homepage = "https://github.com/pengzhendong/pyrnnoise";
      license = [
        lib.licenses.asl20
        lib.licenses.bsd3
      ];
      sourceProvenance = with lib.sourceTypes; [ binaryNativeCode ];
      platforms = [ "x86_64-linux" ];
    };
  };

  onnxruntime = pythonPackages.onnxruntime.overridePythonAttrs (oldAttrs: rec {
    version = "1.24.4";
    src = fetchurl {
      url = "https://files.pythonhosted.org/packages/7f/72/105ec27a78c5aa0154a7c0cd8c41c19a97799c3b12fc30392928997e3be3/onnxruntime-${version}-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
      hash = "sha256-4wyXK8AuBykRqrtokUU+xzeVOGwK8rdhtlREuKTEdF8=";
    };
    unpackPhase = ''
      mkdir dist
      cp "$src" "dist/onnxruntime-${version}-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
    '';
    meta = oldAttrs.meta // {
      sourceProvenance = with lib.sourceTypes; [ binaryNativeCode ];
      platforms = [ "x86_64-linux" ];
    };
  });

  onnxPythonPackages = pythonPackages.overrideScope (_final: previous: {
    protobuf = previous.protobuf6;
  });

  onnx_1_22 = (onnx.override {
    protobuf = protobuf_33;
    python3Packages = onnxPythonPackages;
  }).overrideAttrs (oldAttrs: rec {
    version = "1.22.0";
    src = fetchFromGitHub {
      owner = "onnx";
      repo = "onnx";
      tag = "v${version}";
      hash = "sha256-gc65t/VN3kdvV9tiFoOk6Sw+OZe4Udgm3VcZPP9gzpE=";
    };
    env = oldAttrs.env // {
      BUILD_SHARED_LIBS = "0";
    };
    nativeBuildInputs = oldAttrs.nativeBuildInputs ++ [ onnxPythonPackages.scikit-build-core ];
    postPatch = (oldAttrs.postPatch or "") + ''
      substituteInPlace pyproject.toml \
        --replace-fail '"protobuf==4.25.1"' '"protobuf>=4.25.1"'
    '';
    postInstall = ''
      mkdir -p "$out"
    '';
  });

  pythonOnnx = (pythonPackages.onnx.override {
    onnx = onnx_1_22;
    protobuf = pythonPackages.protobuf6;
  }).overridePythonAttrs (_: {
    enabledTestPaths = [ "onnx/test" ];
  });

  pythonDeps =
    (with pythonPackages; [
      av
      certifi
      click
      filelock
      mediapipe
      numpy
      packaging
      pillow
      psutil
      pygobject3
      pyrnnoise
      protobuf6
      scipy
      setuptools
      urllib3
    ])
    ++ [
      onnxruntime
      pythonOnnx
    ];

  pythonPath = pythonPackages.makePythonPath pythonDeps;
  # Isolated child probes ignore PYTHONPATH. Give their interpreter a genuine
  # Nix-owned dependency prefix instead of weakening the application's -I flag.
  runtimePython = pythonPackages.python.withPackages (_: pythonDeps);
  caBundle = "${pythonPackages.certifi}/${pythonPackages.python.sitePackages}/certifi/cacert.pem";
  emptyCaDirectory = runCommand "nvbroadcast-empty-ca-directory" { } ''mkdir -p "$out"'';
  tlsFixture = fetchurl {
    inherit (qualification.tlsFixture) url;
    sha256 = qualification.tlsFixture.sha256;
  };
  faceModel = fetchurl {
    inherit (qualification.faceModel) url;
    sha256 = qualification.faceModel.sha256;
  };

  v4l-utils-headless = v4l-utils.override {
    withBPF = false;
    withGUI = false;
  };

  requirementsCheck = writeShellScript "nvbroadcast-requirements-check" ''
    if [ -n "''${NVBROADCAST_SKIP_REQUIREMENTS_CHECK:-}" ]; then
      exit 0
    fi

    warn() {
      printf 'nvbroadcast: warning: %s\n' "$*" >&2
    }

    if [ ! -d /sys/module/v4l2loopback ]; then
      warn "v4l2loopback is not loaded; virtual camera output may be unavailable."
    fi

    if ! ${pipewire}/bin/pw-cli info 0 >/dev/null 2>&1; then
      warn "PipeWire does not appear reachable; camera/audio routing may fail."
    fi

    if ! ${pulseaudio}/bin/pactl info >/dev/null 2>&1; then
      warn "PulseAudio-compatible server does not appear reachable; virtual microphone routing may fail."
    fi
  '';
in
assert lib.assertMsg nativeOnly "NV Broadcast supports only native x86_64-linux Nix builds";
assert lib.assertMsg pythonSourcesMatch "Selected and spliced CPython source identities must match the reviewed pin";
pythonPackages.buildPythonApplication (finalAttrs: {
  pname = "nvbroadcast";
  version = "1.5.4";
  pyproject = true;
  __structuredAttrs = true;

  src = lib.fileset.toSource {
    root = ../.;
    # Flake sources can already be in the store (for example, path:/src),
    # where gitTracked cannot inspect Git's index. Keep an explicit list so
    # local, untracked files cannot enter the package source in that case.
    fileset = lib.fileset.unions (
      (map (path: ../. + "/${path}") (
        lib.splitString "\n" (lib.removeSuffix "\n" (builtins.readFile ./source-files.txt))
      )) ++ [
        # Explicit build-test inputs; never installed into the app package.
        ./bootstrap_ci.sh ./hosted_check.py ./qualify_runtime.py
        ./security-inputs.json ./source_files.py
      ]
    );
  };

  build-system = with pythonPackages; [
    setuptools
    wheel
  ];

  dependencies = pythonDeps;

  postInstall = ''
    install -Dm644 LICENSE "$out/share/doc/nvbroadcast/LICENSE"
    for program in nvbroadcast nvbroadcast-vcam; do
      substituteInPlace "$out/bin/$program" \
        --replace-fail '#!${pythonPackages.python.interpreter}' '#!${runtimePython.interpreter}'
    done
  '';

  nativeBuildInputs = [
    gobject-introspection
    makeShellWrapper
    wrapGAppsHook4
  ];

  buildInputs = [
    cairo
    gdk-pixbuf
    glib
    graphene
    gst_all_1.gstreamer
    gst_all_1.gst-plugins-bad
    gst_all_1.gst-plugins-base
    gst_all_1.gst-plugins-good
    gst_all_1.gst-plugins-ugly
    gst_all_1.gst-libav
    gtk4
    libadwaita
    pango
  ];

  nativeCheckInputs = [
    gitMinimal
    pipewire
    psmisc
    pulseaudio
    pythonPackages.pytestCheckHook
    pythonPackages.jsonschema
    v4l-utils-headless
    writableTmpDirAsHomeHook
  ];

  preCheck = ''
    export MPLCONFIGDIR=$TMPDIR/matplotlib
    ${pythonPackages.python.interpreter} -c '
    from importlib.metadata import version
    from packaging.version import Version
    assert version("jsonschema") == "4.26.0"
    assert Version(version("pytest")) >= Version("9.0.3")
    '
  '';

  # Requires real GPU, camera, and v4l2loopback devices.
  disabledTestPaths = [ "tests/test_integration.py" ];

  pythonImportsCheck = [ "nvbroadcast" ];

  dontWrapGApps = true;

  postPatch = ''
    # Python's sys.prefix points at the interpreter, not this application output.
    substituteInPlace src/nvbroadcast/core/resources.py \
      --replace-fail 'Path(sys.prefix) / "share"' \
        'Path("${placeholder "out"}") / "share"'

    # Generated test helpers cannot use /usr/bin/env in the Nix build sandbox.
    substituteInPlace scripts/source_runtime.py \
      --replace-fail '"#!/usr/bin/env bash\nexec "' \
        '"#!${stdenv.shell}\nexec "'
    substituteInPlace tests/test_release_checksums.py \
      --replace-fail '"#!/usr/bin/env python3\n"' \
        '"#!${pythonPackages.python.interpreter}\n"'

    # New installer fixtures also need declared tools in the hermetic sandbox.
    substituteInPlace tests/test_macos_package_install.py \
      --replace-fail '/bin/bash' '${stdenv.shell}' \
      --replace-fail '/bin/cat' '${coreutils}/bin/cat' \
      --replace-fail 'script = script.replace("/bin/ls", str(self.tools / "ls"))' \
        'script = script.replace("/usr/bin/find", "${findutils}/bin/find").replace("/bin/ls", str(self.tools / "ls"))' \
      --replace-fail '{self.tools}:/usr/bin:/bin' \
        '{self.tools}:${builtins.dirOf stdenv.shell}:${lib.makeBinPath [ coreutils ]}:/usr/bin:/bin'
    substituteInPlace tests/test_macos_signing_workflow.py \
      --replace-fail '/bin/bash' '${stdenv.shell}' \
      --replace-fail '"PATH": "/usr/bin:/bin"' \
        '"PATH": "${builtins.dirOf stdenv.shell}:${lib.makeBinPath [ coreutils ]}:/usr/bin:/bin"'

    # Upstream's fixed system PATH is empty in the Nix build sandbox.
    substituteInPlace scripts/native_package_upgrade.sh.in \
      --replace-fail 'PATH=/usr/sbin:/usr/bin:/sbin:/bin' \
        'PATH=${
          lib.makeBinPath [
            coreutils
            gawk
          ]
        }:/usr/sbin:/usr/bin:/sbin:/bin'
  '';

  preFixup = ''
    makeWrapperArgs+=("''${gappsWrapperArgs[@]}")
  '';

  makeWrapperArgs = [
    "--prefix"
    "GST_PLUGIN_SYSTEM_PATH_1_0"
    ":"
    (lib.makeSearchPath "lib/gstreamer-1.0" [
      gst_all_1.gst-plugins-bad
      gst_all_1.gst-plugins-base
      gst_all_1.gst-plugins-good
      gst_all_1.gst-plugins-ugly
      gst_all_1.gst-libav
    ])
    "--prefix"
    "LD_LIBRARY_PATH"
    ":"
    (lib.makeLibraryPath [
      cairo
      glib
      gtk4
      libadwaita
      libglvnd
      libxcb
      stdenv.cc.cc
      zlib
    ])
    "--prefix"
    "PATH"
    ":"
    (lib.makeBinPath [
      pipewire
      psmisc
      pulseaudio
      v4l-utils-headless
    ])
    # Upstream launches child Python processes via sys.executable.
    "--prefix"
    "PYTHONPATH"
    ":"
    "${placeholder "out"}/${pythonPackages.python.sitePackages}:${pythonPath}"
    # Explicit external trust settings remain user configuration. A clean
    # environment uses exactly the current upstream certifi roots, with no
    # implicit older Nix/system capath added by OpenSSL.
    "--set-default" "NIX_SSL_CERT_FILE" caBundle
    "--set-default" "SSL_CERT_FILE" caBundle
    "--set-default" "SSL_CERT_DIR" "${emptyCaDirectory}"
  ];

  postFixup = ''
    mv "$out/bin/nvbroadcast" "$out/bin/.nvbroadcast-gapps-wrapped"
    makeShellWrapper "$out/bin/.nvbroadcast-gapps-wrapped" "$out/bin/nvbroadcast" \
      --run ${requirementsCheck}
  '';

  passthru = {
    pythonSourceQualification = {
      platforms = lib.mapAttrs (_name: platform: { inherit (platform) system config; }) nativePlatforms;
      interpreters = pythonSourceIdentities;
    };
    tests = {
      runtime = runCommand "${finalAttrs.pname}-finished-cpu-runtime-check" { } ''
        mkdir -p "$out" "$TMPDIR/home"
        env -i HOME="$TMPDIR/home" PATH="${lib.makeBinPath [ coreutils ]}" \
          NVBROADCAST_SKIP_REQUIREMENTS_CHECK=1 \
          ${finalAttrs.finalPackage}/bin/nvbroadcast --help > "$out/wrapper-help.txt"
        env -i HOME="$TMPDIR/home" PATH="${lib.makeBinPath [ coreutils ]}" \
          NVBROADCAST_SKIP_REQUIREMENTS_CHECK=1 \
          ${finalAttrs.finalPackage}/bin/nvbroadcast --recording-smoke > "$out/recording.json"
        env -i HOME="$TMPDIR/home" PATH="${lib.makeBinPath [ coreutils ]}" \
          ${runtimePython.interpreter} -I -c '
        import json, mimetypes
        from pathlib import Path
        expected = "${mailcap}/etc/mime.types"
        if mimetypes.knownfiles != [expected] or not Path(expected).is_file():
            raise ValueError("Native mimetypes must use only the selected Nix mailcap database")
        mimetypes.init()
        if mimetypes.guess_type("document.pdf")[0] != "application/pdf":
            raise ValueError("Native mailcap MIME lookup failed")
        print(json.dumps({"knownfiles": mimetypes.knownfiles, "pdf": "application/pdf"}, sort_keys=True))
        ' > "$out/mimetypes.json"
        # Deliberately exclude pytest/check/build environments. These are the
        # finished app's interpreter and its app-only dependency path.
        env -i HOME="$TMPDIR/home" PATH="${lib.makeBinPath [ coreutils ]}" \
          PYTHONNOUSERSITE=1 MPLBACKEND=Agg MPLCONFIGDIR="$TMPDIR/matplotlib" \
          PYTHONPATH="${finalAttrs.finalPackage}/${pythonPackages.python.sitePackages}:${pythonPath}" \
          NIX_SSL_CERT_FILE="${caBundle}" SSL_CERT_FILE="${caBundle}" SSL_CERT_DIR="${emptyCaDirectory}" \
          ${runtimePython.interpreter} ${./qualify_runtime.py} \
          ${finalAttrs.finalPackage} ${finalAttrs.src} ${./security-inputs.json} \
          ${tlsFixture} ${faceModel} "$out/runtime.json" \
          '${builtins.toJSON {
            av = toString av.src;
            audiolab = toString audiolab.src;
            mediapipe = toString mediapipe.src;
            pyrnnoise = toString pyrnnoise.src;
            onnxruntime = toString onnxruntime.src;
            setuptools = toString secureSetuptools.src;
          }}'
      '';

      help = runCommand "${finalAttrs.pname}-help-test" { } ''
        export MPLCONFIGDIR=$TMPDIR/matplotlib
        export NVBROADCAST_SKIP_REQUIREMENTS_CHECK=1
        ${lib.getExe finalAttrs.finalPackage} --help > help.txt
        ${finalAttrs.finalPackage}/bin/nvbroadcast-vcam --help > vcam-help.txt
        grep -F "Show help options" help.txt
        grep -F "Virtual Camera Service" vcam-help.txt
        grep -F 'getattr(b, "_MAX_INFER_HEIGHT", "?")' \
          ${finalAttrs.finalPackage}/${pythonPackages.python.sitePackages}/nvbroadcast/app.py
        PYTHONPATH=${finalAttrs.finalPackage}/${pythonPackages.python.sitePackages}:${pythonPath} \
          ${pythonPackages.python.interpreter} -c '
        import audiolab
        import av
        import mediapipe
        import onnxruntime
        import pyrnnoise
        from google.protobuf import __version__ as protobuf_version
        from packaging.version import Version
        from nvbroadcast.core import dependency_installer, resources
        assert av.__version__ == "16.0.1"
        assert onnxruntime.__version__ == "1.24.4"
        assert Version("6.33.5") <= Version(protobuf_version) < Version("7")
        assert dependency_installer._runtime_install_block_reason() is not None
        assert resources.find_app_icon().is_file()
        assert resources.find_app_icon_png().is_file()
        assert resources.find_backgrounds_dir().is_dir()
        '
        touch $out
      '';

    };
  };

  meta = {
    description = "AI-powered virtual camera with background removal and noise cancellation";
    longDescription = ''
      NV Broadcast — Unofficial NVIDIA Broadcast for Linux and other OS.

      AI-powered virtual camera with background removal, blur, replacement,
      video enhancement, and noise cancellation. This flake currently packages
      the CPU inference runtime. Open source.
    '';
    homepage = "https://github.com/Hkshoonya/nvidia-broadcast-linux";
    changelog = "https://github.com/Hkshoonya/nvidia-broadcast-linux/releases/tag/v${finalAttrs.version}";
    license = lib.licenses.gpl3Plus;
    maintainers = with lib.maintainers; [ Tenshock ];
    mainProgram = "nvbroadcast";
    platforms = [ "x86_64-linux" ];
  };
})
