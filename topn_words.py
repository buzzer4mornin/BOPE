#! /usr/bin/python

# by_AYDIN:
# usage: python topn_words.py ./data/nyt_voca.txt ./models/nyt/4list_tops.txt topn_output_aydin.txt
# usage2: python topn_words.py ./data/ap_voca.txt ./models/nyt/list_tops.txt ap_topn_output_aydin.txt
# usage2: python topn_words.py ./data/ctmp_vocab.txt ./models/ctmp/list_tops.txt ctmp_topn_output_aydin.txt
# blei topics: http://www.cs.columbia.edu/~blei/lda-c/ap-topics.pdf
import sys
import math


def print_topics(vocab_file, nwords, result_file):
    with open(vocab_file, 'r') as f:
        vocab = f.readlines()

    vocab = list(map(lambda x: x.strip(), vocab))
    vocab_index = {i: w for i, w in zip(range(len(vocab)),vocab)}

    with open(nwords, "r") as n:
        lines = n.readlines()
        for l in lines:
            l = l.split()
            converts = list(map(lambda x: vocab_index[int(x)], l))
            converts = " ".join(converts)
            with open (result_file, "a") as r:
                r.write(converts + "\n")



if (__name__ == '__main__'):
    vocab_file = sys.argv[1]
    nwords = sys.argv[2]
    result_file = sys.argv[3]
    print_topics(vocab_file, nwords, result_file)
