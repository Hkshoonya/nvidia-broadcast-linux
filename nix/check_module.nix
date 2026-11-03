{ self, nixpkgs, pkgs }:
let
  lib = nixpkgs.lib;
  make = system: overrides: (lib.nixosSystem {
    inherit system;
    modules = [
      self.nixosModules.default
      ({ ... }: { system.stateVersion = "25.05"; })
      overrides
    ];
  }).config;
  defaults = make "x86_64-linux" { programs.nvbroadcast.enable = true; };
  customized = make "x86_64-linux" {
    programs.nvbroadcast.enable = true;
    programs.nvbroadcast.v4l2loopback = {
      videoNr = 42;
      cardLabel = "NVIDIA \"quoted\" \\ Broadcast";
      exclusiveCaps = false;
      maxBuffers = 8;
    };
  };
  minimal = make "x86_64-linux" {
    programs.nvbroadcast.enable = true;
    programs.nvbroadcast.pipewire.enable = false;
    programs.nvbroadcast.v4l2loopback.enable = false;
  };
  disabled = make "x86_64-linux" { };
  nvidia = make "x86_64-linux" {
    programs.nvbroadcast.enable = true;
    programs.nvbroadcast.nvidia.enable = true;
  };
  arm = make "aarch64-linux" { programs.nvbroadcast.enable = true; };
  results = {
    defaultPackage = builtins.elem defaults.programs.nvbroadcast.package defaults.environment.systemPackages;
    defaultPipewire = defaults.services.pipewire.enable && defaults.services.pipewire.pulse.enable;
    defaultLoopback = builtins.elem "v4l2loopback" defaults.boot.kernelModules;
    defaultOptions = lib.hasInfix
      "options v4l2loopback devices=1 video_nr=10 card_label=\"NVbroadcast\" exclusive_caps=1 max_buffers=4"
      defaults.boot.extraModprobeConfig;
    customizedOptions = lib.hasInfix
      ''video_nr=42 card_label="NVIDIA \"quoted\" \\ Broadcast" exclusive_caps=0 max_buffers=8''
      customized.boot.extraModprobeConfig;
    minimalPackage = builtins.elem minimal.programs.nvbroadcast.package minimal.environment.systemPackages;
    minimalLoopbackDisabled = !(builtins.elem "v4l2loopback" minimal.boot.kernelModules);
    minimalPipewireDisabled = !minimal.services.pipewire.enable;
    disabledPackage = !(builtins.any (package: (package.pname or "") == "nvbroadcast") disabled.environment.systemPackages);
    disabledLoopback = !(builtins.elem "v4l2loopback" disabled.boot.kernelModules);
    nvidiaGraphics = nvidia.hardware.graphics.enable && nvidia.services.xserver.videoDrivers == [ "nvidia" ];
    unsupportedArchitecture = builtins.any
      (assertion: !assertion.assertion && lib.hasInfix "x86_64-linux" assertion.message)
      arm.assertions;
  };
in
assert lib.assertMsg (builtins.all (value: value) (builtins.attrValues results))
  "NV Broadcast NixOS module configuration evaluation failed";
pkgs.writeText "nvbroadcast-module-evaluation.json" (builtins.toJSON {
  scope = "Configuration evaluation only; no NixOS boot or device acceptance";
  inherit results;
})
