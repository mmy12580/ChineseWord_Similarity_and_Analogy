import argparse
import numpy as np 
import multiprocessing as mp

from utils import cosine
from utils import rho
from utils import load_embedding


class TestAnalogy(object):
    
    def __init__(self, test_file, embd_vec):
       self.calcAnalogy(test_file, embd_vec)


    def findIdx4word(self, word, sim_res):
        """Finding
        
        Arguments:
            word {str} -- A true word in analogy test file
            sim_res {list} -- A list of similar results
        
        Returns:
            int -- Index of ranking results 
            int -- Indicator if the true word in the list
        """
        for idx, item in enumerate(sim_res):
            if word == item:
                return idx, 1
        return None, 0

    def calcAnalogy(self, test_file, embd_vec):
        """A function caculate word analogy 
        
        Arguments:
            test_file {str} -- A local path for embedding
            embd_vec {str} -- Human designed analogy file
        """
        res_dict = {}
        ranks = []
        num = 0
        right_pred = 0
        prev_topic = ''
        with open(test_file, 'r', encoding='utf-8') as fr:
            for i, line in enumerate(fr):
                if i == 0:
                    prev_topic = line.split()[1].split('-')[0]
                    continue

                if ":" in line:
                    topic = line.split()[1].split('-')[0]
                    res_dict[prev_topic] = {'count': num,
                                            'acc': right_pred/num, 
                                            'average_ranks': np.nanmean(ranks)}
                    num = 0
                    right_pred = 0
                    prev_topic = topic
                else:
                    words = line.strip('\n').split()
                    if len(words) != 4:
                        continue 

                    if any([w not in embd_vec for w in words]):
                        print(f'Line {i} have word(s) not in given embedding')
                        continue

                    result = embd_vec.most_similar(positive=words[1::-1], negative=[words[-1]])
                    rank, pred = self.findIdx4word(words[-2], result)
                    ranks.append(rank)
                    right_pred += pred
                    num += 1
                
        for key, val in res_dict.items():
            print('Category: ', key)
            print('Number found: ', val['count'])
            print('Accuracy: ', val['acc'])
            print('Mean Rank: ', val['average_ranks'])


if __name__ == "__main__":

    parser = argparse.ArgumentParser('Word Anology Evaluation')
    parser.add_argument('--embd_path', type=str, default=None,
                        help='Embedding file e.g word2vec, glove, fasttext')
    parser.add_argument('--test_file', type=str, default=None,
                        help='Human designed word analogy test file.')
    args = parser.parse_args()

    embedding = load_embedding(args.embd_path)
    TestAnalogy(args.test_file, embedding)

