{
  lib,
  stdenvNoCC,
  mdbook-mermaid,
  mdbook-linkcheck,

  # custom args
  rustPackages,
  rustPackageNames,
}:
stdenvNoCC.mkDerivation {
  __structuredAttrs = true;

  name = "psyche-book";
  src = ./.;

  nativeBuildInputs = [
    mdbook-mermaid
    mdbook-linkcheck
  ];

  postPatch = ''
    mkdir -p generated/cli

    # we set HOME to a writable directory to avoid cache dir permission issues
    export HOME=$TMPDIR

    ${lib.concatMapStringsSep "\n" (
      name:
      let
        noPythonPackage = "${name}-nopython";
      in
      "${rustPackages.${noPythonPackage}}/bin/${name} print-all-help --markdown > generated/cli/${
        lib.replaceStrings [ "-" ] [ "-" ] name
      }.md"
    ) rustPackageNames}

    cp ${../secrets.nix} generated/secrets.nix
  '';

  buildPhase = "mdbook build";

  installPhase = ''
    mkdir -p $out
    cp -r book/html/* $out/
  '';
}
