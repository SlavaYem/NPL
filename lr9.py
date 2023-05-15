from nltk.book import *
from nltk.corpus import nps_chat
import nltk

# fdist1 = FreqDist(text1)
# print(fdist1)
# print(fdist1.most_common(50))
# print(fdist1['whale'])
# print(fdist1.hapaxes())
# V = set(text1)
# long_words = [w for w in V if len(w) > 15]
# print(sorted(long_words))
# fdist5 = FreqDist(text5)
# print(sorted(w for w in set(text5) if len(w) > 7 and fdist5[w] > 7))
# print(list(bigrams(['more', 'is', 'said', 'than', 'done'])))
# print(text4.collocations())
# print(text8.collocations())
# print([len(w) for w in text1])
# fdist = FreqDist(len(w) for w in text1)
# print(fdist)
# print(fdist.most_common())
# print(fdist.max())
# print(fdist[3])
# print(fdist.freq(3))
# print(sent7)
# print([w for w in sent7 if len(w) < 4])
# print([w for w in sent7 if len(w) < 4])
# print(sorted(term for term in set(text4) if 'gnt' in term))
# print(sorted(w for w in set(text7) if '-' in w and 'index' in w))
# print(sorted(wd for wd in set(text3) if wd.istitle() and len(wd) > 10))
# print(sorted(w for w in set(sent7) if not w.islower()))
# print(sorted(t for t in set(text2) if 'cie' in t or 'cei' in t))
# word = 'cat'
# if len(word) < 5:
#     print('word length is less than 5')
# //...1...//
chat_words = nps_chat.words()
b_words = [word for word in chat_words if word.startswith('b')]
b_words_sorted = sorted(b_words)
sorted_list = []
for word in b_words_sorted:
    sorted_list.append(word)
print(sorted_list)
# //...2...//
print(list(range(10)),
      list(range(10, 20)),
      list(range(10, 20, 2)),
      list(range(20, 10, -2)))
# //...3...//
sunset_index = text9.index("sunset")
print("Index of 'sunset':", sunset_index)
start_index = sunset_index - 10
end_index = sunset_index + 10
sentence_slice = text9[start_index:end_index]
print("Complete sentence:", " ".join(sentence_slice))
# //...4...//
sentences = sent1 + sent2 + sent3 + sent4 + sent5 + sent6 + sent7 + sent8
vocabulary = set(sentences)
sorted_vocabulary = sorted(vocabulary)
print(sorted_vocabulary)
# //...5...//
print(sorted(set(w.lower() for w in text1)))
print(sorted(w.lower() for w in set(text1)))
# //...6...//
# w.isupper()
# not w.islower()
# //...7...//
last_two_words = text2[-2:]
print(last_two_words)
# //...8...//
chat_words = nps_chat.words()
four_letter_words = [word.lower() for word in chat_words if len(word) == 4 and word.isalpha()]
freq_dist = nltk.FreqDist(four_letter_words)
for word, frequency in freq_dist.most_common():
    print(word, frequency)
# //...9...//
upper_list = []
for word in text6:
    if word.isupper():
        upper_list.append(word)
print(upper_list)
# //...10...//
ending_in_ize = [word for word in text6 if word.endswith('ize')]
print(ending_in_ize)
containing_z = [word for word in text6 if 'z' in word]
print(containing_z)
containing_pt = [word for word in text6 if 'pt' in word]
print(containing_pt)
titlecase_words = [word for word in text6 if word.istitle()]
print(titlecase_words)
# //...11...//
sent = ['she', 'sells', 'sea', 'shells', 'by', 'the', 'sea', 'shore']
sh_words = [word for word in sent if word.startswith('sh')]
print("Words beginning with 'sh':", sh_words)
long_words = [word for word in sent if len(word) > 4]
print("Words longer than four characters:", long_words)
# //...12...//
total_word_length = sum(len(w) for w in text1)
total_words = len(text1)
average_word_length = total_word_length / total_words
print("Average word length:", average_word_length)


# //...13...//


def vocab_size(text):
    unique_words = set(text)
    return len(unique_words)


size = vocab_size(text1)
print("Vocabulary size:", size)


# //...14...//


def percent(word, text):
    word_count = text.count(word)
    percentage = (word_count / len(text)) * 100
    return percentage


word = "example"
percentage = percent(word, text1)
print(f"The word '{word}' appears in the text with a percentage of {percentage:.2f}%")
# //...15...//
print(set(sent3) < set(text1))
