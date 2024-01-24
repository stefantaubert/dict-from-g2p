from collections import OrderedDict

from ordered_set import OrderedSet

from dict_from_g2pE.core import transcribe_with_g2pE


def test_component():
  vocabulary = OrderedSet((
    "Test?",
    "Test-def.",
    "abc,",
    "\"def",
  ))

  result_dict = transcribe_with_g2pE(vocabulary, trim=set("?,\"."))

  assert result_dict == OrderedDict([('Test?', OrderedDict([(('T', 'EH1', 'S', 'T', '?'), 1.0)])), ('Test-def.', OrderedDict(
    [(('T', 'EH1', 'S', 'T', '-', 'D', 'EH1', 'F', '.'), 1.0)])), ('abc,', OrderedDict([(('AE1', 'B', 'K', ','), 1.0)])), ('"def', OrderedDict([(('"', 'D', 'EH1', 'F'), 1.0)]))])
