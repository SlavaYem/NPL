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
print("###################### Task 1 ######################")
num_words = len(text2)
print("Number of words in text2:", num_words)
num_distinct_words = len(set(text2))
print("Number of distinct words in text2:", num_distinct_words)

# //...2...//
print("###################### Task 2 ######################")


def lexical_diversity(text):
    return len(set(text)) / len(text)


humor_ld = lexical_diversity(text1)
romance_ld = lexical_diversity(text2)

print("Lexical diversity score for humor fiction:", humor_ld)
print("Lexical diversity score for romance fiction:", romance_ld)
# //...3...//
print("###################### Task 3 ######################")
text2.dispersion_plot(["Elinor", "Marianne", "Edward", "Willoughby"])
# //...4...//
print("###################### Task 4 ######################")
print(text5.collocations())
# //...5...//
print("###################### Task 5 ######################")
set(text4)
print(len(set(text4)))
# //...6...//
print("###################### Task 6 ######################")
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
# //...7...//
print("###################### Task 7 ######################")
my_sent = ["Слава", "Україні"]
my_string = ' '.join(my_sent)
print(my_string)
back_to_list = my_string.split()
print(back_to_list)
# //...8...//
print("###################### Task 8 ######################")
phrase1 = ["Слава"]
phrase2 = ["Україні"]
phrase3 = ["Героям"]
phrase4 = ["Слава"]
sentence1 = phrase1 + phrase2
sentence2 = phrase3 + phrase4
sentence3 = phrase1 + phrase2 + phrase3 + phrase4
print(sentence1)
print(len(sentence1))
print(sentence2)
print(len(sentence2))
print(sentence3)
print(len(sentence3))
print(len(phrase1 + phrase2))
print(len(phrase1) + len(phrase2))
# //...9...//
print("###################### Task 9 ######################")
print("Monty Python"[6:12])
print(["Monty", "Python"][1])
# //...10...//
print("###################### Task 10 ######################")
print(sent1[2][2])
# //...11...//
print("###################### Task 11 ######################")
sent3 = ['In', 'the', 'beginning', 'God', 'created', 'the', 'heaven', 'and', 'the', 'earth', '.']
indexes = []
for i, word in enumerate(sent3):
    if word == 'the':
        indexes.append(i)
print(indexes)
