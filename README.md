# dict-from-g2pE

[![PyPI](https://img.shields.io/pypi/v/dict-from-g2pE.svg)](https://pypi.python.org/pypi/dict-from-g2pE)
[![PyPI](https://img.shields.io/pypi/pyversions/dict-from-g2pE.svg)](https://pypi.python.org/pypi/dict-from-g2pE)
[![MIT](https://img.shields.io/github/license/stefantaubert/dict-from-g2p.svg)](LICENSE)

CLI to create a pronunciation dictionary by predicting English ARPAbet phonemes using seq2seq model from [g2pE](https://www.github.com/kyubyong/g2p) and the possibility of ignoring punctuation and splitting on hyphens before prediction.

## Installation

```sh
pip install dict-from-g2pE --user
```

## Usage

```sh
dict-from-g2pE-cli
```

### Example

```sh
# Create example vocabulary
cat > /tmp/vocabulary.txt << EOF
Test?
abc,
"def
Test-def.
"xyz?
"uv-w?
EOF

# Create dictionary from vocabulary and example dictionary
dict-from-g2pE-cli \
  /tmp/vocabulary.txt \
  /tmp/result.dict \
  --split-on-hyphen \
  --n-jobs 4

cat /tmp/result.dict
# -------
# Output:
# -------
# Test?  T EH1 S T ?
# abc,  AE1 B K ,
# "def  " D EH1 F
# Test-def.  T EH1 S T - D EH1 F .
# "xyz?  " Z IH1 JH IH0 Z ?
# "uv-w?  " AH1 V - V IY1 ?
# -------
```

## Acknowledgments

[g2pE: A Simple Python Module for English Grapheme To Phoneme Conversion](https://www.github.com/kyubyong/g2p)
