from argparse import ArgumentParser, Namespace
from logging import getLogger

from ordered_set import OrderedSet
from pronunciation_dictionary import SerializationOptions, save_dict

from dict_from_g2pE.argparse_helper import (DEFAULT_PUNCTUATION, ConvertToOrderedSetAction,
                                            add_chunksize_argument, add_encoding_argument,
                                            add_maxtaskperchild_argument, add_n_jobs_argument,
                                            add_serialization_group, parse_existing_file,
                                            parse_non_empty_or_whitespace, parse_path,
                                            parse_positive_float)
from dict_from_g2pE.core import transcribe_with_g2pE


def get_app_try_add_vocabulary_from_pronunciations_parser(parser: ArgumentParser):
  parser.description = "Transcribe vocabulary using g2p."
  # TODO support multiple files
  parser.add_argument("vocabulary", metavar='vocabulary', type=parse_existing_file,
                      help="file containing the vocabulary (words separated by line)")
  add_encoding_argument(parser, "--vocabulary-encoding", "encoding of vocabulary")
  parser.add_argument("dictionary", metavar='dictionary', type=parse_path,
                      help="path to output created dictionary")
  parser.add_argument("--weight", type=parse_positive_float,
                      help="weight to assign for each pronunciation", default=1.0)
  parser.add_argument("--trim", type=parse_non_empty_or_whitespace, metavar='SYMBOL', nargs='*',
                      help="trim these symbols from the start and end of a word before lookup", action=ConvertToOrderedSetAction, default=DEFAULT_PUNCTUATION)
  parser.add_argument("--split-on-hyphen", action="store_true",
                      help="split words on hyphen symbol before lookup")
  add_serialization_group(parser)
  mp_group = parser.add_argument_group("multiprocessing arguments")
  add_n_jobs_argument(mp_group)
  add_chunksize_argument(mp_group)
  add_maxtaskperchild_argument(mp_group)
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
  dictionary_instance = transcribe_with_g2pE(
    vocabulary_words, ns.weight, ns.trim, ns.split_on_hyphen, ns.n_jobs, ns.maxtasksperchild, ns.chunksize, silent=False)

  s_options = SerializationOptions(ns.parts_sep, ns.include_numbers, ns.include_weights)

  try:
    save_dict(dictionary_instance, ns.dictionary, ns.serialization_encoding, s_options)
  except Exception as ex:
    logger.error("Dictionary couldn't be written.")
    logger.debug(ex)
    return False

  logger.info(f"Written dictionary to: {ns.dictionary.absolute()}")

  return True
