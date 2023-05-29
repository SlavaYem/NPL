import computer as computer
import nltk
from nltk import FreqDist
import random
from nltk.book import *

nltk.download('gutenberg')
from nltk.corpus import gutenberg
from nltk.corpus import brown
from nltk.corpus import webtext
from nltk.corpus import state_union
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from nltk.corpus import udhr

print("###################### Task 1 ######################")
phrase = ["Hello", "world", "I", "am", "Slava"]
new_phrase = phrase + ["Nice", "to", "meet", "you"]
print(new_phrase)
repeated_phrase = phrase * 3
print(repeated_phrase)
print(phrase[0])
print(phrase[-1])
sub_phrase = phrase[1:4]
print(sub_phrase)
sorted_phrase = sorted(phrase)
print(sorted_phrase)
print("###################### Task 2 ######################")
words = gutenberg.words('austen-persuasion.txt')
num_tokens = len(words)
num_types = len(set(words))
print("Total word tokens: ", num_tokens)
print("Total word types: ", num_types)
print("###################### Task 3 ######################")
brown_samples = nltk.corpus.brown.words(categories=['news', 'romance'])
print("Sample text from the Brown corpus:")
print(brown_samples[:100])
webtext_samples = nltk.corpus.webtext.words()
print("\nSample text from the Web text corpus:")
print(webtext_samples[:100])
print("###################### Task 4 ######################")
state_union_files = nltk.corpus.state_union.fileids()
men_counts = {}
women_counts = {}
people_counts = {}
for fileid in state_union_files:
    address_text = nltk.corpus.state_union.words(fileid)
    men_counts[fileid] = address_text.count('men')
    women_counts[fileid] = address_text.count('women')
    people_counts[fileid] = address_text.count('people')
print("Year\tMen\tWomen\tPeople")
for fileid in state_union_files:
    year = fileid[:4]
    men_count = men_counts[fileid]
    women_count = women_counts[fileid]
    people_count = people_counts[fileid]
    print(f"{year}\t{men_count}\t{women_count}\t{people_count}")
print("###################### Task 5 ######################")
print(wn.synset('tree.n.01').member_meronyms())
print(wn.synset('tree.n.01').part_meronyms())
print(wn.synset('tree.n.01').substance_meronyms())
print(wn.synset('tree.n.01').member_holonyms())
print(wn.synset('tree.n.01').part_holonyms())
print(wn.synset('tree.n.01').substance_holonyms())
print("###################### Task 7 ######################")
corpus = nltk.corpus.gutenberg
text = corpus.raw('austen-persuasion.txt')
tokens = nltk.word_tokenize(text)
text_obj = nltk.Text(tokens)
text_obj.concordance("however")
print("###################### Task 8 ######################")
names = nltk.corpus.names
cfd = nltk.ConditionalFreqDist(
    (fileid, name[0])
    for fileid in names.fileids()
    for name in names.words(fileid))
cfd.plot()
print("###################### Task 9 ######################")
news_text = brown.words(categories='news')
romance_text = brown.words(categories='romance')
print("Vocabulary of news:  ", len(set(news_text)))
print("Vocabulary of romance:", len(set(romance_text)))
print("---------------------------")
print("Vocabulary richness of news:\t", len(set(news_text)) / len(news_text))
print("Vocabulary richness of romance:\t", len(set(romance_text)) / len(romance_text))
print("----------------------------------------------------")
print("'Address' in news:")
nltk.Text(news_text).similar('address')
print()
print("'Address' in romance:")
nltk.Text(romance_text).similar('address')
print("###################### Task 11 ######################")
modals = ['can', 'could', 'may', 'might', 'must', 'will']
cfd = nltk.ConditionalFreqDist(
    (genre, word.lower())
    for genre in brown.categories()
    for word in brown.words(categories=genre))
genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']
cfd.tabulate(conditions=genres, samples=modals)
print("###################### Task 12 ######################")
prondict = nltk.corpus.cmudict.dict()
print('Distinct words:', len(prondict))
wordPron = 0
for key in prondict:
    if len(prondict[key]) > 1:
        wordPron += 1
print('Fractions of words with more than one possible pronunciation:', wordPron / len(prondict))
print("###################### Task 13 ######################")
noun_synsets = len(list(wn.all_synsets('n')))
cnt = 0
for synset in wn.all_synsets('n'):
    if (synset.hyponyms() == []):
        cnt += 1
print(cnt / noun_synsets)
print("###################### Task 14 ######################")


def supergloss(synset):
    definitions = [synset.definition()]
    hypernyms = synset.hypernyms()
    for hypernym in hypernyms:
        definitions.append(hypernym.definition())
    hyponyms = synset.hyponyms()
    for hyponym in hyponyms:
        definitions.append(hyponym.definition())

    return ' '.join(definitions)


synset = wn.synsets('computer')[0]
result = supergloss(synset)
print(result)
print("###################### Task 15 ######################")
wordSet = []  #
fdist = FreqDist(w.lower() for w in brown.words() if w.isalpha())
for sample in fdist:
    if fdist[sample] >= 3:
        wordSet.append(sample)
print(wordSet)
print("###################### Task 16 ######################")
for category in brown.categories():
    tokens = len(brown.words(categories=category))
    types = len(set(brown.words(categories=category)))
    diversity = types / tokens
    print(category, diversity)
print("###################### Task 17 ######################")


def find_50_most_frequent_words(text):
    fdist = FreqDist(w.lower() for w in text if w.isalpha() and w.lower() not in stopwords.words('english'))
    return fdist.most_common(50)


print(find_50_most_frequent_words(text1))
print("###################### Task 18 ######################")


def find_50_most_frequent_bigrams(text):
    bigram = list(nltk.bigrams(text))
    fdist = FreqDist(b for b in bigram if b[0].isalpha() and b[1].isalpha()
                     and b[0] not in stopwords.words('english')
                     and b[1] not in stopwords.words('english'))
    return fdist.most_common(50)


print(find_50_most_frequent_bigrams(text1))
print("###################### Task 19 ######################")
cfd = nltk.ConditionalFreqDist(
    (genre, word)
    for genre in brown.categories()
    for word in brown.words(categories=genre))
genres = brown.categories()
my_words = ['love', 'like', 'peace', 'hate', 'war', 'fight', 'battle']
cfd.tabulate(conditions=genres, samples=my_words)
print("###################### Task 20 ######################")


def word_freq(section):
    fdist = FreqDist(w.lower() for w in brown.words(categories=section))
    return fdist


print(word_freq('news'))
print("###################### Task 21 ######################")


def number_of_syllables(text):
    prondict = nltk.corpus.cmudict.dict()
    number = 0
    for w in text:
        if w.lower() in prondict.keys():
            number += len(prondict[w.lower()][0])
    return number


print(number_of_syllables(text1))
print("###################### Task 22 ######################")


def hedge(text):
    new_version = list(text)
    for i in range(2, len(text) + len(text) // 3, 3):
        new_version.insert(i, 'like')
    return nltk.Text(new_version)


print(hedge(text1))
print("###################### Task 23 ######################")


def zipf_law(text):
    fdist = FreqDist([w.lower() for w in text if w.isalpha()])
    fdist = fdist.most_common()
    rank = []
    freq = []
    n = 1
    for i in range(len(fdist)):
        freq.append(fdist[i][1])
        rank.append(n)
        n += 1
    plt.plot(rank, freq, 'bs')
    plt.xscale('log')
    plt.title("Zipf's law")
    plt.xlabel('word rank')
    plt.ylabel('word frequency')
    plt.show()


zipf_law(brown.words())
print("###################### Task 24 ######################")


def generate_random_text_on_n_most_likely_words(text, n):
    fdist = FreqDist(text)
    fdist = fdist.most_common(n)
    for i in range(n):
        print(random.choice(fdist)[0], end=' ')


generate_random_text_on_n_most_likely_words(text1, 200)
generate_random_text_on_n_most_likely_words(brown.words(categories='news'), 200)
generate_random_text_on_n_most_likely_words(brown.words(categories=['news', 'romance']), 200)
print("###################### Task 25 ######################")


def find_language(s):
    langs = []
    for lang in udhr.fileids():
        if lang.endswith('Latin1') and s in udhr.words(lang):
            langs.append(lang)
    return langs


print(find_language('world'))
print("###################### Task 26 ######################")
cnt = 0
hypos = 0
for synset in wn.all_synsets('n'):
    if synset.hyponyms() != []:
        hypos += len(synset.hyponyms())
        cnt += 1
print(hypos / cnt)
print("###################### Task 27 ######################")


def average_polysemy(pos):
    count = 0
    total_senses = 0

    for synset in wn.all_synsets(pos):
        count += 1
        total_senses += len(synset.lemmas())

    return total_senses / count


average_noun_polysemy = average_polysemy(wn.NOUN)
average_verb_polysemy = average_polysemy(wn.VERB)
average_adj_polysemy = average_polysemy(wn.ADJ)
average_adv_polysemy = average_polysemy(wn.ADV)
print("Average Polysemy:")
print("Nouns:", average_noun_polysemy)
print("Verbs:", average_verb_polysemy)
print("Adjectives:", average_adj_polysemy)
print("Adverbs:", average_adv_polysemy)
print("###################### Task 28 ######################")


def similarities(w1, w2):
    print('Path similarity:', w1.path_similarity(w2))
    print('Leacock-Chodorow similarity:', w1.lch_similarity(w2))
    print('Wu-Palmer similarity:', w1.wup_similarity(w2))

    def similarities(w1, w2):
        print('Path similarity:', w1.path_similarity(w2))
        print('Leacock-Chodorow similarity:', w1.lch_similarity(w2))
        print('Wu-Palmer similarity:', w1.wup_similarity(w2))


car = wn.synset('car.n.01')
automobile = wn.synset('automobile.n.01')
gem = wn.synset('gem.n.01')
jewel = wn.synset('jewel.n.01')
journey = wn.synset('journey.n.01')
voyage = wn.synset('voyage.n.01')
boy = wn.synset('boy.n.01')
lad = wn.synset('lad.n.01')
coast = wn.synset('coast.n.01')
shore = wn.synset('shore.n.01')
asylum = wn.synset('asylum.n.01')
madhouse = wn.synset('madhouse.n.01')
magician = wn.synset('magician.n.01')
wizard = wn.synset('wizard.n.01')
midday = wn.synset('midday.n.01')
noon = wn.synset('noon.n.01')
furnace = wn.synset('furnace.n.01')
stove = wn.synset('stove.n.01')
food = wn.synset('food.n.01')
fruit = wn.synset('fruit.n.01')
bird = wn.synset('bird.n.01')
cock = wn.synset('cock.n.01')
crane = wn.synset('crane.n.01')
tool = wn.synset('tool.n.01')
implement = wn.synset('implement.n.01')
brother = wn.synset('brother.n.01')
monk = wn.synset('monk.n.01')
oracle = wn.synset('oracle.n.01')
cemetery = wn.synset('cemetery.n.01')
woodland = wn.synset('woodland.n.01')
rooster = wn.synset('rooster.n.01')
hill = wn.synset('hill.n.01')
forest = wn.synset('forest.n.01')
graveyard = wn.synset('graveyard.n.01')
slave = wn.synset('slave.n.01')
chord = wn.synset('chord.n.01')
smile = wn.synset('smile.n.01')
glass = wn.synset('glass.n.01')
string = wn.synset('string.n.01')
similarities(forest, glass)
