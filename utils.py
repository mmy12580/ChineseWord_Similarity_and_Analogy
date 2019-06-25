import os 
import numpy as numpy
from scipy import spatial
from scipy.stats import spearmanr
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile


def cosine(vec1, vec2):
    """Cosine score
       Reference: https://en.wikipedia.org/wiki/Cosine_similarity
    """
    return 1 - spatial.distance.cosine(vec1, vec2)


def rho(vec1, vec2):
    """Spearman's rho
       Reference: https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient
    """
    return spearmanr(vec1, vec2)[0]


def load_embedding(embd_path, is_glove=False):
    """A wrapper function loads embedding from any kind.
    
    Arguments:
        embd_path {str} -- Pre-trained embedding path
        is_glove {bool} -- Whether it is glove embedding
    """
    if is_glove:
        glove_file = datapath(embd_path)
        tmp_file = get_tmpfile('word2vec.txt') 
        glove2word2vec(glove_file, tmp_file)
        return KeyedVectors.load_word2vec_format(tmp_file)
    
    if embd_path.endswith('bin'):
        return KeyedVectors.load_word2vec_format(embd_path, binary=True)    
    return KeyedVectors.load_word2vec_format(embd_path, binary=False)