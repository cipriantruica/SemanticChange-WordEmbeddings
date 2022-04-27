# coding: utf-8

__author__      = "Ciprian-Octavian Truică"
__copyright__   = "Copyright 2022, University Politehnica of Bucharest"
__license__     = "GNU GPL"
__version__     = "0.1"
__email__       = "ciprian.truica@upb.ro"
__status__      = "Development"

from gensim.models import Word2Vec, FastText
from gensim import corpora

import numpy as np

class WordEmbeddings:
    ###########################################
    # corpus (list of lists of lists) - The corpus is a list that contain for each document a list of sentences. Each sentece is a list of words (see example in main)
    # window_size (int) – The maximum distance between the current and predicted word within a sentence.
    # vector_size  (int) – Dimensionality of the word vectors.
    # learning_rate (float) – Learning rate will linearly drop to learning_rate as training progresses.
    # epochs (int) – Number of epochs for training.
    # workers (int) – Use these many worker threads to train the model (=faster training with multicore machines).
    ###########################################
    def __init__(self, corpus, window_size=10, vector_size =128, learning_rate=0.05, epochs=100, workers=4):
        self.corpus = corpus
        self.no_docs = len(self.corpus)
        self.window_size = window_size
        self.vector_size = vector_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.workers = workers

        # Initialize auxiliary variables (some are required for other tasks, i.e., classification)
        self.documents = []
        self.sentences = []
        self.word2id = {}
        self.no_words = 0 # total number of words
        self.max_size = 0 # max size of largest document

        #  initialize the models
        self.w2v_cbow_model = None
        self.w2v_sg_model = None
        self.ft_cbow_model = None
        self.ft_sg_model = None
        


    def preprareDocuments(self):
        word_id = 1
        for document in self.corpus:
            doc = []
            for sentence in document:
                self.sentences.append(sentence)
                for word in sentence:
                    if self.word2id.get(word) is None:
                        self.word2id[word] = word_id
                        word_id += 1
                    doc.append(self.word2id[word])
            if self.max_size < len(doc):
                self.max_size = len(doc)
            self.documents.append(doc)
        
        self.no_words = len(self.word2id) + 1
        
        return np.array(self.documents)
        

    #####################
    # min_count (int, optional) – Ignores all words with total frequency lower than this.
    # sg ({0, 1}, optional) – Training algorithm: 1 for skip-gram; otherwise CBOW.
    # hs ({0, 1}, optional) – If 1, hierarchical softmax will be used for model training. If 0, and negative is non-zero, negative sampling will be used.
    # cbow_mean ({0, 1}, optional) – If 0, use the sum of the context word vectors. If 1, use the mean, only applies when cbow is used.
    #####################
    def word2vecEmbedding(self, min_count=1, sg=0, hs=0, cbow_mean=1):
        self.word2vec = np.empty(shape=(self.no_words, self.vector_size))
        model = Word2Vec(self.sentences, vector_size=self.vector_size, window=self.window_size, min_count=min_count, workers=self.workers, sg=sg, hs=hs, cbow_mean=cbow_mean, alpha=self.learning_rate, epochs=self.epochs)

        self.word2vec[0] = np.array([0] * self.vector_size)
        for word in self.word2id:
            self.word2vec[self.word2id[word]] = np.array(model.wv[word])
        if sg == 0:
            self.w2v_cbow_model = model
        else:
            self.w2v_sg_model = model

        return self.word2vec


    #####################
    # min_count (int, optional) – Ignores all words with total frequency lower than this.
    # sg ({0, 1}, optional) – Training algorithm: 1 for skip-gram; otherwise CBOW.
    # hs ({0, 1}, optional) – If 1, hierarchical softmax will be used for model training. If 0, and negative is non-zero, negative sampling will be used.
    # cbow_mean ({0, 1}, optional) – If 0, use the sum of the context word vectors. If 1, use the mean, only applies when cbow is used.
    #####################
    def word2FastTextEmbeddings(self, min_count=1, sg=0, hs=0, cbow_mean=1):
        self.word2fasttext = np.empty(shape=(self.no_words, self.vector_size))
        model = FastText(self.sentences, vector_size=self.vector_size, window=self.window_size, min_count=min_count, workers=self.workers, sg=sg, hs=hs, cbow_mean=cbow_mean, alpha=self.learning_rate, epochs=self.epochs)

        # the first vector contains only 0 (this is required for 0-padding documents)
        self.word2fasttext[0] = np.array([0] * self.vector_size)
        for word in self.word2id:
            self.word2fasttext[self.word2id[word]] = np.array(model.wv[word])


        if sg == 0:
            self.ft_cbow_model = model
        else:
            self.ft_sg_model = model

        return self.word2fasttext

    
if __name__ == '__main__':
    # a list of document
    documents = [
        "A language family is a group of languages related through descent from a common ancestral language or parental language, called the proto-language of that family. The term \"family\" reflects the tree model of language origination in historical linguistics, which makes use of a metaphor comparing languages to people in a biological family tree, or in a subsequent modification, to species in a phylogenetic tree of evolutionary taxonomy. Linguists therefore describe the daughter languages within a language family as being genetically related.",
        "According to Ethnologue there are 7,139 living human languages distributed in 142 different language families. A living language is defined as one that is the first language of at least one person. There are also many dead languages, or languages which have no native speakers living, and extinct languages, which have no native speakers and no descendant languages. Finally, there are some languages that are insufficiently studied to be classified, and probably some which are not even known to exist outside their respective speech communities.",
        "Membership of languages in a language family is established by research in comparative linguistics. Sister languages are said to descend \"genetically\" from a common ancestor. Speakers of a language family belong to a common speech community. The divergence of a proto-language into daughter languages typically occurs through geographical separation, with the original speech community gradually evolving into distinct linguistic units. Individuals belonging to other speech communities may also adopt languages from a different language family through the language shift process.",
        "Genealogically related languages present shared retentions; that is, features of the proto-language (or reflexes of such features) that cannot be explained by chance or borrowing (convergence). Membership in a branch or group within a language family is established by shared innovations; that is, common features of those languages that are not found in the common ancestor of the entire family. For example, Germanic languages are \"Germanic\" in that they share vocabulary and grammatical features that are not believed to have been present in the Proto-Indo-European language. These features are believed to be innovations that took place in Proto-Germanic, a descendant of Proto-Indo-European that was the source of all Germanic languages."
    ]

    # (Simple) Preprocessing steps:
    # 1. split each document intro sentences
    # 2. extract tokens
    # Note: for more advanced analysis, implement your own advanced preprocessing pipeline.

    from nltk.tokenize import sent_tokenize, word_tokenize
    corpus = []
    for document in documents:
        doc = []
        for sentence in sent_tokenize(document):
            doc.append(word_tokenize(sentence))
        corpus.append(doc)

    print(corpus)
    # The corpus is a list documents. Each document is a list of sentences. Each sentece is a list of tokens.
    # The corpus looks like this:
    # [
    #    # 1st document
    #    [ 
    #       [token_1, token_2, ..., token_m ], # 1st sentence of the 1st document 
    #       [token_1, token_2, ..., token_m ], # 2nd sentence of the 1st document
    #       ...
    #       [token_1, token_2, ..., token_m ], # n-th sentence of the 1st document
    #    ],
    #    # 2nd document
    #    [ 
    #       [token_1, token_2, ..., token_m ], # 1st sentence of the 2nd document 
    #       [token_1, token_2, ..., token_m ], # 2nd sentence of the 2nd document
    #       ...
    #       [token_1, token_2, ..., token_m ], # n-th sentence of the 2nd document
    #    ],
    #    ...
    #    # k-th document
    #    [ 
    #       [token_1, token_2, ..., token_m ], # 1st sentence of the k-th document 
    #       [token_1, token_2, ..., token_m ], # 2nd sentence of the k-th document
    #       ...
    #       [token_1, token_2, ..., token_m ], # n-th sentence of the k-th document
    #    ],
    # ]

    we = WordEmbeddings(corpus)
    # Prepare the corpus for word embeddings
    we.preprareDocuments()
    print("Word to ID mapping:")
    for word in we.word2id:
        print("Word: {} ID: {}".format(word, we.word2id[word]))

    # Get Word2Vec CBOW embeddings
    w2v_cbow = we.word2vecEmbedding(sg=0)
    print("\nWord2Vec CBOW matrix\n", w2v_cbow)
    
    # Get Word2Vec Skip-Gram embeddings
    w2v_sg = we.word2vecEmbedding(sg=1)
    print("\nWord2Vec CBOW matrix\n", w2v_sg)

    # Get FastText CBOW embeddings
    w2f_cbow = we.word2FastTextEmbeddings(sg=0)
    print("\nFastText CBOW matrix\n", w2f_cbow)

    # Get Word2Vec Skip-Gram embeddings
    w2f_sg = we.word2FastTextEmbeddings(sg=1)
    print("\nFastText Skip-Gram matrix\n", w2f_sg)
    
    print("\n\n")
    # Get vectors for word "languages"
    print("\nWord2Vec CBOW for word language\n", w2v_cbow[we.word2id['languages']])
    print("\nWord2Vec Skip-Gram for word language\n", w2v_sg[we.word2id['languages']])
    print("\nFastText CBOW for word language\n", w2f_cbow[we.word2id['languages']])
    print("\nFastText Skip-Gram for word language\n", w2f_sg[we.word2id['languages']])
