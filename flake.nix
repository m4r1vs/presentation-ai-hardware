{
  description = "Flake to get Typst running with fonts installed";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
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
            fontDirectories = [
              eb-garamond
              (stdenv.mkDerivation {
                name = "Apple Color Emoji Font";
                src = fetchurl {
                  url = "https://github.com/samuelngs/apple-emoji-linux/releases/latest/download/AppleColorEmoji.ttf";
                  hash = "sha256-SG3JQLybhY/fMX+XqmB/BKhQSBB0N1VRqa+H6laVUPE=";
                };
                dontUnpack = true;
                installPhase = ''
                  runHook preInstall

                  mkdir -p $out/share/fonts/truetype
                  cp $src $out/share/fonts/truetype/AppleColorEmoji.ttf

                  runHook postInstall
                '';
              })
            ];
          };
          buildInputs = [
            typst
            typstyle
            tinymist
          ];
        };
    });
}
