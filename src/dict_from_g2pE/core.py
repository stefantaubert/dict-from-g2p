from collections import OrderedDict
from functools import partial
from logging import getLogger
from multiprocessing.pool import Pool
from typing import Any, Dict, Optional, Set, Tuple

import nltk
from ordered_set import OrderedSet
from pronunciation_dictionary import PronunciationDict, Pronunciations, Word
from tqdm import tqdm
from word_to_pronunciation import Options, get_pronunciations_from_word


def transcribe_with_g2pE(vocabulary: OrderedSet[Word], *, weight: float = 1.0, trim: Optional[Set[str]] = None, split_on_hyphen: bool = True, n_jobs: int = 4, maxtasksperchild: Optional[int] = None, chunksize: int = 100_000, silent: bool = False):
  if trim is None:
    trim = set()
  trim_symbols = ''.join(trim)
  options = Options(trim_symbols, split_on_hyphen, False, False, 1.0)
  dictionary_instance = get_pronunciations(
    vocabulary, weight, options, n_jobs, maxtasksperchild, chunksize, silent)
  return dictionary_instance


def download_nltk_data():
  logger = getLogger(__name__)
  try:
    nltk.data.find('taggers/averaged_perceptron_tagger.zip')
  except LookupError:
    logger.info("Downloading 'averaged_perceptron_tagger' from nltk ...")
    nltk.download('averaged_perceptron_tagger')
  try:
    nltk.data.find('corpora/cmudict.zip')
  except LookupError:
    logger.info("Downloading 'cmudict' from nltk ...")
    nltk.download('cmudict')


def get_pronunciations(vocabulary: OrderedSet[Word], weight: float, options: Options, n_jobs: int, maxtasksperchild: Optional[int], chunksize: int, silent: bool) -> PronunciationDict:
  lookup_method = partial(
    process_get_pronunciation,
    weight=weight,
    options=options,
  )

  download_nltk_data()
  # G2p downloads NLTK data on import which could take a while
  # pylint: disable=import-outside-toplevel
  from g2p_en import G2p
  model = G2p()

  with Pool(
    processes=n_jobs,
    initializer=__init_pool_prepare_cache_mp,
    initargs=(vocabulary, model),
    maxtasksperchild=maxtasksperchild,
  ) as pool:
    entries = range(len(vocabulary))
    iterator = pool.imap(lookup_method, entries, chunksize)
    pronunciations_to_i = dict(tqdm(iterator, total=len(entries), unit="words", disable=silent))

  return get_dictionary(pronunciations_to_i, vocabulary)


def get_dictionary(pronunciations_to_i: Dict[int, Pronunciations], vocabulary: OrderedSet[Word]) -> PronunciationDict:
  resulting_dict = OrderedDict()

  for i, word in enumerate(vocabulary):
    pronunciations = pronunciations_to_i[i]
    assert len(pronunciations) == 1
    assert word not in resulting_dict
    resulting_dict[word] = pronunciations

  return resulting_dict


process_unique_words: OrderedSet[Word] = None
process_model: Any = None


def __init_pool_prepare_cache_mp(words: OrderedSet[Word], model: Any) -> None:
  global process_unique_words
  global process_model
  process_unique_words = words
  process_model = model


def process_get_pronunciation(word_i: int, weight: float, options: Options) -> Tuple[int, Pronunciations]:
  global process_unique_words
  global process_model
  assert 0 <= word_i < len(process_unique_words)
  word = process_unique_words[word_i]

  # TODO support all entries; also create all combinations with hyphen then
  lookup_method = partial(
    lookup_in_model,
    model=process_model,
    weight=weight,
  )

  pronunciations = get_pronunciations_from_word(word, lookup_method, options)
  # logger = getLogger(__name__)
  # logger.debug(pronunciations)
  return word_i, pronunciations


def lookup_in_model(word: Word, model: Any, weight: float) -> Pronunciations:
  assert len(word) > 0
  # lower() because G2p seems to predict only lower-case words correctly
  word = word.lower()
  result = model.predict(word)
  result = tuple(result)
  result = OrderedDict((
    (result, weight),
  ))
  return result
