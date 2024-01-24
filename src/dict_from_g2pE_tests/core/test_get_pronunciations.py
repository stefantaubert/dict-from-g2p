from collections import OrderedDict

from ordered_set import OrderedSet
from word_to_pronunciation import Options

from dict_from_g2pE.core import get_pronunciations


def test_component():
  vocabulary = OrderedSet((
    "Test?",
    "Test-def.",
    "abc,",
    "\"def",
  ))
  options = Options("?,\".", True, False, False, 1.0)

  result_dict = get_pronunciations(vocabulary, 1.0, options, 1, None, 4, silent=True)

  assert result_dict == OrderedDict([('Test?', OrderedDict([(('T', 'EH1', 'S', 'T', '?'), 1.0)])), ('Test-def.', OrderedDict(
    [(('T', 'EH1', 'S', 'T', '-', 'D', 'EH1', 'F', '.'), 1.0)])), ('abc,', OrderedDict([(('AE1', 'B', 'K', ','), 1.0)])), ('"def', OrderedDict([(('"', 'D', 'EH1', 'F'), 1.0)]))])
