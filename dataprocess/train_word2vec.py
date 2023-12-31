import re
import os, string

import time
from gensim.models import Word2Vec
from Config.ConfigT import MyConf
def train_word2vec(all_data):
    # Word2vec parameters
    min_word_frequency_word2vec = 5
    embed_size_word2vec = 200
    context_window_word2vec = 5
    # Learn the word2vec model and extract vocabulary
    wordvec_path = os.path.join('../resources/word2vec/',
        '{}features_{}minwords_{}context.model'.format(embed_size_word2vec, min_word_frequency_word2vec,
                                                       context_window_word2vec))
    print('starts loading model.....')
    start = time.time()
    if os.path.exists(wordvec_path):
        # load word2vec model
        wordvec_model = Word2Vec.load(wordvec_path)
    else:
        wordvec_model = Word2Vec(all_data, min_count=min_word_frequency_word2vec, size=embed_size_word2vec,
                                 window=context_window_word2vec)
        wordvec_model.save(wordvec_path)
    end = time.time()
    print('Finish load, Time-consuming%s s' % (end - start))


def t_word2vec():
    word2vec_model = Word2Vec.load(config.WordEmbeddingPath)
    sims=word2vec_model.most_similar('bug',topn=10)
    print(sims)



if __name__ == '__main__':
    # train word2vec
    config=MyConf('../Config/config.cfg')
    data=[]
    # with open(r'F:\PyProject\DuplicateBugReportDetect\resources\corpus\merge_corpus.txt') as f:
    #     for line in f.readlines():
    #         data.append(line.strip('\n').split(' '))
    # train_word2vec(data)

    # test word2vc
    t_word2vec()