from gensim.models.keyedvectors import KeyedVectors
import word2vec
import timeit


start_time = timeit.default_timer()

word2vec.word2phrase('3g-p.txt', '3g-phrases.txt', verbose=True)
word2vec.word2vec('3g-phrases.txt', '3g.bin', size=100, verbose=True)

elapsed = timeit.default_timer() - start_time
InMinutes = elapsed / 60

word2vec.word2clusters('3g-p.txt', '3g-clusters.txt', 100, verbose=True)

model = KeyedVectors.load_word2vec_format('3g.bin', binary=True)

model.save_word2vec_format('3g-vectors.txt', binary=False)

print ("The Totatl Execution Time in Minutes is: ", InMinutes)
