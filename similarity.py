import argparse
from utils import cosine
from utils import rho
from utils import load_embedding


class TestSimilarty(object):
    
    def __init__(self, test_file, embd_vec):
        self.calc_word_similarity(test_file, embd_vec)

    def calc_word_similarity(self, test_file, embed_vec):
        """Calculate Word Similarity
        
        Arguments:
            test_file {str} -- similarity test file e.g wordsim-240 and wordsim-297
            embed_vec {Keyedvectors} -- A pre-load gensim word vectors
        """
        pred, label, found = [], [], 0
        with open(test_file, 'r', encoding='utf8') as fr:
            lines = fr.readlines()
            for line in lines:
                w1, w2, score = line.split()
                if w1 in embed_vec and w2 in embed_vec:
                    found += 1
                    pred.append(cosine(embed_vec[w1], embed_vec[w2]))
                    label.append(float(score))

        file_name = test_file.split("/")[-1].replace('.txt', '')
        print(f"Test File: {file_name}")
        print(f"Numbers of words Found: {found}") 
        print(f"Numbers of words Not Found: {len(lines) - found}")
        print(f"Spearman's Rank Coeficient: {rho(label, pred)}") 


if __name__ == "__main__":

    parser = argparse.ArgumentParser('Word Similarity Evaluation')
    parser.add_argument('--embd_path', type=str, default=None,
                        help='Embedding file e.g word2vec, glove, fasttext')
    parser.add_argument('--test_file', type=str, default=None,
                        help='Human designed word similarity test file.')
    args = parser.parse_args()

    embedding = load_embedding(args.embd_path)
    TestSimilarty(args.test_file, embedding)