#!/bin/bash

prog_name="dict-from-g2p-cli"
cli_path=src/dict_from_g2p/cli.py

mkdir -p ./dist

pipenv run cxfreeze \
  -O \
  --compress \
  --target-dir=dist \
  --bin-includes "libffi.so" \
  --target-name=cli \
  $cli_path

cd dist
zip $prog_name-linux.zip ./ -r
cd ..
echo "zipped."