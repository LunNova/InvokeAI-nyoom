let
  pkgs = import <nixpkgs> { };
  python = pkgs.python310;
  pythonWithPackages = (python.withPackages (
    # TODO: way more packages
    python-packages: with python-packages; [
      pip
      wheel
      setuptools # for pkg_resources
      types-setuptools

      python-dotenv
      pillow
      numpy
      httpx
      aiofiles
      tqdm
      h2
    ]
  ));
in
pkgs.mkShell {
  nativeBuildInputs = [
    pkgs.pkg-config
  ];
  buildInputs = [
    pkgs.bashInteractive
    pythonWithPackages
  ];
  # https://nixos.wiki/wiki/Python#Emulating_virtualenv_with_nix-shell
  shellHook = ''
    # Tells pip to put packages into $PIP_PREFIX instead of the usual locations.
    # See https://pip.pypa.io/en/stable/user_guide/#environment-variables.
    export PIP_PREFIX=$(pwd)/venv
    export PYTHONPATH="$(pwd):$PIP_PREFIX/${python.sitePackages}:$PYTHONPATH"
    export PATH="$PIP_PREFIX/bin:$PATH"
    unset SOURCE_DATE_EPOCH

    mkdir -p $(pwd)/venv
    ln -sfn ${pythonWithPackages} $(pwd)/venv/nix-shell-python

    #python setup.py egg_info
    #pip install `grep -v '^\[' *.egg-info/requires.txt` || true
  '';
}
