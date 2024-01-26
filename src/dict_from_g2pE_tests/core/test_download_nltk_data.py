import os

import nltk

from dict_from_g2pE.core import download_nltk_data


def test_component():
  try:
    path = nltk.data.find('taggers/averaged_perceptron_tagger.zip')
    os.remove(path.zipfile.filename)
  except LookupError:
    pass

  try:
    path = nltk.data.find('corpora/cmudict.zip')
    os.remove(path.zipfile.filename)
  except LookupError:
    pass

  download_nltk_data()

  nltk.data.find('taggers/averaged_perceptron_tagger.zip')
  nltk.data.find('corpora/cmudict.zip')
