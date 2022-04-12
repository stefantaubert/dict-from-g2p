from g2p_en import G2p
from collections import OrderedDict
from argparse import ArgumentParser, Namespace
from logging import getLogger
from tqdm import tqdm
from functools import partial
from multiprocessing.pool import Pool
from typing import Dict, Optional, Tuple
from pronunciation_dictionary import PronunciationDict, Word, Pronunciations, save_dict_to_file, SerializationOptions
from word_to_pronunciation import get_pronunciations_from_word, Options
from ordered_set import OrderedSet
from dict_from_g2p.argparse_helper import ConvertToOrderedSetAction, DEFAULT_PUNCTUATION, add_chunksize_argument, add_encoding_argument, add_maxtaskperchild_argument, add_n_jobs_argument, add_serialization_group, parse_existing_file, parse_non_empty_or_whitespace, parse_path, parse_positive_float


def get_app_try_add_vocabulary_from_pronunciations_parser(parser: ArgumentParser):
  parser.description = "Transcribe vocabulary using g2p."
  # todo support multiple files
  parser.add_argument("vocabulary", metavar='vocabulary', type=parse_existing_file,
                      help="file containing the vocabulary (words separated by line)")
  parser.add_argument("dictionary", metavar='dictionary', type=parse_path,
                      help="path to output created dictionary")
  parser.add_argument("--weight", type=parse_positive_float,
                      help="weight to assign for each annotation", default=1.0)
  parser.add_argument("--trim", type=parse_non_empty_or_whitespace, metavar='SYMBOL', nargs='*',
                      help="trim these symbols from the start and end of a word before looking it up in the reference pronunciation dictionary", action=ConvertToOrderedSetAction, default=DEFAULT_PUNCTUATION)
  parser.add_argument("--split-on-hyphen", action="store_true",
                      help="split words on hyphen symbol before lookup")
  add_encoding_argument(parser, "--vocabulary-encoding", "encoding of vocabulary")
  add_serialization_group(parser)
  add_n_jobs_argument(parser)
  add_chunksize_argument(parser)
  add_maxtaskperchild_argument(parser)
  return get_pronunciations_files


def get_pronunciations_files(ns: Namespace) -> bool:
  assert ns.vocabulary.is_file()
  logger = getLogger(__name__)

  try:
    vocabulary_content = ns.vocabulary.read_text(ns.vocabulary_encoding)
  except Exception as ex:
    logger.error("Vocabulary couldn't be read.")
    return False

  vocabulary_words = OrderedSet(vocabulary_content.splitlines())
  trim_symbols = ''.join(ns.trim)
  options = Options(trim_symbols, ns.split_on_hyphen, False, False, 1.0)

  dictionary_instance = get_pronunciations(
    vocabulary_words, ns.weight, options, ns.n_jobs, ns.maxtasksperchild, ns.chunksize)

  s_options = SerializationOptions(ns.parts_sep, ns.include_numbers, ns.include_weights)

  try:
    save_dict_to_file(dictionary_instance, ns.dictionary, ns.serialization_encoding, s_options)
  except Exception as ex:
    logger.error("Dictionary couldn't be written.")
    logger.debug(ex)
    return False

  logger.info(f"Written dictionary to: {ns.dictionary.absolute()}")

  return True


def get_pronunciations(vocabulary: OrderedSet[Word], weight: float, options: Options, n_jobs: int, maxtasksperchild: Optional[int], chunksize: int) -> PronunciationDict:
  lookup_method = partial(
    process_get_pronunciation,
    weight=weight,
    options=options,
  )

  model = G2p()

  with Pool(
    processes=n_jobs,
    initializer=__init_pool_prepare_cache_mp,
    initargs=(vocabulary, model),
    maxtasksperchild=maxtasksperchild,
  ) as pool:
    entries = range(len(vocabulary))
    iterator = pool.imap(lookup_method, entries, chunksize)
    pronunciations_to_i = dict(tqdm(iterator, total=len(entries), unit="words"))

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
process_model: G2p = None


def __init_pool_prepare_cache_mp(words: OrderedSet[Word], model: G2p) -> None:
  global process_unique_words
  global process_model
  process_unique_words = words
  process_model = model


def process_get_pronunciation(word_i: int, weight: float, options: Options) -> Tuple[int, Pronunciations]:
  global process_unique_words
  assert 0 <= word_i < len(process_unique_words)
  word = process_unique_words[word_i]

  # TODO support all entries; also create all combinations with hyphen then
  lookup_method = partial(
    lookup_in_model,
    weight=weight,
  )

  pronunciations = get_pronunciations_from_word(word, lookup_method, options)

  return word_i, pronunciations


def lookup_in_model(word: Word, weight: float) -> Pronunciations:
  global process_model
  result = process_model.predict(word)
  result = OrderedDict((
    (result, weight),
  ))
  return result
