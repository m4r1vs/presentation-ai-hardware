{
  description = "A very basic flake";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/master";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = {
    nixpkgs,
    flake-utils,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {
        inherit system;
      };
    in {
      devShell = with pkgs;
        mkShell {
          FONTCONFIG_FILE = makeFontsConf {
            fontDirectories = [eb-garamond];
          };
          buildInputs = [
            typst
            typstyle
            tinymist
          ];
        };
    });
}
