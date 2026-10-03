{
  description = "NV Broadcast for Linux";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
      nvbroadcast = pkgs.callPackage ./nix/package.nix { };
    in
    {
      packages.${system} = {
        inherit nvbroadcast;
        default = nvbroadcast;
      };

      checks.${system} = {
        inherit nvbroadcast;
        inherit (nvbroadcast.passthru.tests) help;
      };

      formatter.${system} = pkgs.nixfmt-tree;

      nixosModules = rec {
        nvbroadcast = import ./nix/module.nix {
          package = self.packages.${system}.nvbroadcast;
        };
        default = nvbroadcast;
      };
    };
}
