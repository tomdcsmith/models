import argparse
import pickle
import sys

from gensim.models import Word2Vec


def main():
    parser = ArgParser(sys.argv)
    vocab_list = parse_vocab(parser.vocab_file)
    word_vecs = get_word_vecs(parser.word2vec_file, vocab_list)
    pickle.dump(word_vecs, open(parser.output_file, "wb"))


def get_word_vecs(wordvec_file, vocab_list):
    word2vec_model = Word2Vec.load_word2vec_format(wordvec_file, binary=True)
    word_vecs = {}
    for word in vocab_list:
        if word in word2vec_model:
            word_vecs[word] = word2vec_model[word]

    return word_vecs


def parse_vocab(vocab_file):
    vocab_set = set()

    with open(vocab_file, 'r') as vocab_f:
        for line in vocab_f:
            pieces = line.split()
            if len(pieces) != 2:
                sys.stderr.write('Bad line: %s\n' % line)
                continue
            vocab_set.add(pieces[0])

    return list(vocab_set)


class ArgParser:
    def __init__(self, argv):
        parser = argparse.ArgumentParser()
        parser.add_argument("--word2vec_file", dest="word2vec_file", required=True)
        parser.add_argument("--output_file", dest="output_file", required=True)
        parser.add_argument("--vocab_file", dest="vocab_file", required=True)
        args = parser.parse_args(args=argv[1:])
        self.word2vec_file = args.word2vec_file
        self.output_file = args.output_file
        self.vocab_file = args.vocab_file


if __name__ == '__main__':
    main()
