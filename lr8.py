from nltk.book import *
from nltk.corpus import gutenberg
from nltk.draw import dispersion_plot
import matplotlib.pyplot as plt
import numpy as np

#
# text4.dispersion_plot(["citizens", "democracy", "freedom", "duties", "America"])
#
#
# def lexical_diversity(text):
#     return print(len(set(text)) / len(text))
#
#
# def percentage(count, total):
#     return print(100 * count / total)
#
#
# percentage(4, 5)
# percentage(text4.count('a'), len(text4))
# lexical_diversity(text5)
#
# sent1 = ['Call', 'me', 'Ishmael', '.']
# print(len(sent1))
# lexical_diversity(sent1)
# print(sent2, sent3)
# ex1 = ['Monty', 'Python', 'and', 'the', 'Holy', 'Grail']
# print(sorted(ex1), len(ex1), ex1.count('Python'))
# print(['Monty', 'Python'] + ['and', 'the', 'Holy', 'Grail'])
# print(sent4 + sent1)
# sent1.append("Some")
# print(sent1)
# print(text4[173], text4.index('awaken'))
# print(text5[16715:16735])
# sent = ['word1', 'word2', 'word3', 'word4', 'word5',
#         'word6', 'word7', 'word8', 'word9', 'word10']
# print(sent[0], sent[9])
# sent[0] = 'First'
# sent[9] = 'Last'
# print(len(sent))
# print(sent[1:9])
# //...1...//
num_words = len(text2)
print("Number of words in text2:", num_words)
num_distinct_words = len(set(text2))
print("Number of distinct words in text2:", num_distinct_words)


# //...2...//


def lexical_diversity(text):
    return len(set(text)) / len(text)


humor_ld = lexical_diversity(text1)
romance_ld = lexical_diversity(text2)

print("Lexical diversity score for humor fiction:", humor_ld)
print("Lexical diversity score for romance fiction:", romance_ld)
# //...3...//
# sense = gutenberg.words('austen-sense.txt')
# elinor = ["Elinor"]
# marianne = ["Marianne"]
# edward = ["Edward"]
# willoughby = ["Willoughby"]
# for i in range(len(sense)):
#     if sense[i] == "Elinor":
#         elinor.append(i)
#     elif sense[i] == "Marianne":
#         marianne.append(i)
#     elif sense[i] == "Edward":
#         edward.append(i)
#     elif sense[i] == "Willoughby":
#         willoughby.append(i)
# dispersion_plot(sense, elinor, marianne, edward, willoughby)
# //...4...//
print(text5.collocations())
# //...5...//
set(text4)
print(len(set(text4)))
# //...6...//
my_string = "The quick brown fox jumps over the lazy dog"
print(my_string)
concatenated = my_string + my_string
print(concatenated)
multiplied = my_string * 3
print(multiplied)
with_spaces = my_string + " " + my_string
print(with_spaces)
with_spaces = (my_string + " ") * 3
print(with_spaces)
# //...6...//
