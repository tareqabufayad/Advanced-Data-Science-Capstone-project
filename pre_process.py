# -*- coding: utf-8 -*-
import re
import codecs
import pyarabic.araby as araby
import timeit


def read_file():
    f = codecs.open("3g.txt", "r", encoding='utf8')
    data = f.read()
    f.close()
    return data


def write_file(data):
    filew = codecs.open('3g-p.txt', 'w', encoding='utf8')
    filew.write(data)

    filew.close()


def write_one_col():
    with open('10-wiki.txt', 'r') as f, open('one-col.txt', 'w') as f2:
        for line in f:
            for word in line.split():
                f2.write(word + '\n')


def normailize_data(data):
    regex = ur'[\u0621-\u063A\u0641-\u064A]+'
    return " ".join(re.findall(regex, data))


def strip_tatweel(text):

    reduced = araby.strip_tatweel(text)
    return reduced


def strip_tashkeel(text):
    reduced = araby.strip_tashkeel(text)
    return reduced


start_time = timeit.default_timer()
data = read_file()
remove_tashkeel = strip_tashkeel(data)
remove_tatweel = strip_tatweel(remove_tashkeel)
normailized = normailize_data(remove_tatweel)
write_file(normailized)

elapsed = timeit.default_timer() - start_time
InMinutes = elapsed / 60

print ("The Totatl Execution Time in Minutes is: ", InMinutes)