import nltk
import datetime

from nltk import RegexpTagger
from nltk.corpus import brown
import matplotlib.pyplot as plt
from nltk import ConfusionMatrix
from collections import defaultdict
from collections import defaultdict
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
from nltk.tag import UnigramTagger
from nltk.tag import DefaultTagger
from nltk.tag import UnigramTagger, BigramTagger, TrigramTagger
from nltk.tag import brill, BrillTaggerTrainer
from nltk.tag.brill import SymmetricProximateTokensTemplate, ProximateTokensTemplate
from nltk.tag import NgramTagger, DefaultTagger, UnigramTagger

print("###################### Task 1 ######################")
headline = 'Juvenile/NOUN Court/NOUN to/PRT Try/VERB Shooting/ADJ Defendant/NOUN'
print([nltk.tag.str2tuple(t) for t in headline.split()])
print("###################### Task 3 ######################")
sent = 'They wind back the clock, while we chase after the wind.'
print(nltk.pos_tag(nltk.word_tokenize(sent)))
print("###################### Task 7 ######################")
d1 = {'hello': 1, 'world': 2, 'natural': 0}
d2 = {'natural': 3, 'language': 4, 'processing': 5}
d1.update(d2)
print(d1)
print("###################### Task 8 ######################")
e = {}
e['headword'] = ['NOUN', 'a word or term placed at the beginning (as of a chapter or an entry in an encyclopedia)']
e['part-of-speech'] = ['PHRASE',
                       'a traditional class of words distinguished according to the kind of idea denoted and the function performed in a sentence']
e['sense'] = ['NOUN', 'a meaning conveyed or intended']
e['example'] = ['NOUN', 'one that serves as a pattern to be imitated or not to be imitated']
print("###################### Task 10 ######################")
brown_tagged_sents = brown.tagged_sents(categories='news')
unigram_tagger = nltk.UnigramTagger(brown_tagged_sents)
test_text = ['hello', 'world', 'natural', 'language', 'processing']
print(unigram_tagger.tag(test_text))
print("###################### Task 11 ######################")
affix_tagger = nltk.AffixTagger(brown_tagged_sents, affix_length=3, min_stem_length=4)
test_text = 'Experiment with different settings for the affix length and the minimum word length'.split()
print(affix_tagger.tag(test_text))
print("###################### Task 13 ######################")
print(datetime.datetime.today().strftime("%Y-%m-%d"))
print("###################### Task 14 ######################")
list_of_tags = sorted(set([tag for (_, tag) in brown.tagged_words()]))
print(list_of_tags)
print("###################### Task 15 ######################")
brown_tagged = brown.tagged_words()
cfd = nltk.ConditionalFreqDist(brown_tagged)
common_plural = set()
for word in set(brown.words()):
    if cfd[word + 's']['NNS'] > cfd[word]['NN']:
        common_plural.add(word)
tag_dict = {k: len(cfd[k]) for k in cfd}
greatest = max(tag_dict, key=lambda key: tag_dict[key])
helper_list = [t for (_, t) in brown_tagged]
fd = nltk.FreqDist(helper_list)
print(fd.most_common(20))
word_tag_pairs = nltk.bigrams(brown_tagged)
noun_after = [b[1] for (a, b) in word_tag_pairs if a[1].startswith('NN')]
fdist = nltk.FreqDist(noun_after)
print([tag for (tag, _) in fdist.most_common(10)])
print("###################### Task 18 ######################")
brown_tag = brown.tagged_words(tagset='universal')
cfd = nltk.ConditionalFreqDist(brown_tag)
proportion = sum(1 for word in cfd if len(cfd[word]) == 1) / len(cfd)
ambiguous = sum(1 for word in cfd if len(cfd[word]) > 1)
print(proportion)
print(ambiguous)
print("###################### Task 20 ######################")
nltk.download('brown')
md_words = sorted(set(word.lower() for (word, tag) in brown.tagged_words() if tag == 'MD'))
print(md_words)
plural_nouns_verbs = sorted(set(word.lower() for (word, tag) in brown.tagged_words()
                                if tag in ['NNS', 'VBZ'] and word.isalpha()))
print(plural_nouns_verbs)
prepositional_phrases = []
for sent in brown.tagged_sents():
    for i in range(len(sent) - 2):
        if sent[i][1] == 'IN' and sent[i + 1][1] == 'DT' and sent[i + 2][1] == 'NN':
            phrase = ' '.join(word.lower() for (word, _) in sent[i:i + 3])
            prepositional_phrases.append(phrase)
prepositional_phrases = sorted(set(prepositional_phrases))
print(prepositional_phrases)
masculine_pronouns = len(
    [word for (word, tag) in brown.tagged_words() if tag == 'PP' and word.lower() in ['he', 'him', 'his']])
feminine_pronouns = len(
    [word for (word, tag) in brown.tagged_words() if tag == 'PP' and word.lower() in ['she', 'her', 'hers']])

if feminine_pronouns > 0:
    ratio = masculine_pronouns / feminine_pronouns
    print("Masculine pronouns:", masculine_pronouns)
    print("Feminine pronouns:", feminine_pronouns)
    print("Ratio of masculine to feminine pronouns:", ratio)
else:
    print("No feminine pronouns found in the Brown Corpus.")
print("###################### Task 21 ######################")
verbs = ['adore', 'love', 'like', 'prefer']
adverbs = []
for verb in verbs:
    verb_adverbs = []
    for sent in brown.tagged_sents():
        for i in range(1, len(sent) - 1):
            if sent[i][0].lower() == verb:
                if sent[i - 1][1] == 'RB':
                    verb_adverbs.append(sent[i - 1][0].lower())
    adverbs.extend(verb_adverbs)
adverbs = sorted(set(adverbs))
print(adverbs)
print("###################### Task 22 ######################")
patterns = [
    (r'.*s$', 'NNS'),
    (r'.*ing$', 'VBG'),
    (r'.*ed$', 'VBD'),
    (r'^[A-Z][a-z]+$', 'NNP'),
    (r'.*ly$', 'RB')
]
regexp_tagger = RegexpTagger(patterns)
print(regexp_tagger)
print("###################### Task 23 ######################")
corpus = brown.tagged_sents(categories='news')
size = int(len(corpus) * 0.8)
train_sents = corpus[:size]
test_sents = corpus[size:]
patterns = [
    (r'.*s$', 'NNS'),
    (r'.*ing$', 'VBG'),
    (r'.*ed$', 'VBD'),
    (r'^[A-Z][a-z]+$', 'NNP'),
    (r'.*ly$', 'RB')
]

regexp_tagger = nltk.RegexpTagger(patterns)
accuracy = regexp_tagger.evaluate(test_sents)
print("Accuracy:", accuracy)
print("###################### Task 24 ######################")
corpus = brown.tagged_sents(categories='news')
size = int(len(corpus) * 0.8)
train_sents = corpus[:size]
test_sents = corpus[size:]
vocabulary_size = 105
tagset_size = 102
table = []
for n in range(1, 7):
    ngram_tagger = nltk.NgramTagger(n, train_sents)
    accuracy = ngram_tagger.evaluate(test_sents)
    table.append((n, accuracy))
print("n\tAccuracy")
print("----------------")
for n, accuracy in table:
    print(f"{n}\t{accuracy}")
print("###################### Task 26 ######################")
training_sizes = [100, 500, 1000, 2000, 4000, 8000, 16000]
corpus = brown.tagged_sents(categories='news')
accuracies = []
for size in training_sizes:
    train_sents = corpus[:size]
    test_sents = corpus[500:1000]
    unigram_tagger = nltk.UnigramTagger(train_sents)
    accuracy = unigram_tagger.evaluate(test_sents)
    accuracies.append(accuracy)
plt.plot(training_sizes, accuracies)
plt.xlabel('Training Data Size')
plt.ylabel('Accuracy')
plt.title('Performance of Unigram Tagger')
plt.show()
print("###################### Task 27 ######################")
train_sents = brown.tagged_sents(categories='news')[:500]
t2 = nltk.BigramTagger(train_sents)
test_sents = brown.tagged_sents(categories='news', tagset='universal')[500:1000]
filtered_test_sents = [sent for sent in test_sents if len(sent) > 0]
tagged_sents = t2.tag_sents([[word for word, _ in sent] for sent in filtered_test_sents])
gold_tags = [tag for sent in filtered_test_sents for _, tag in sent]
predicted_tags = [tag for sent in tagged_sents for _, tag in sent]
gold_tags = [tag for tag in gold_tags if tag is not None]
predicted_tags = [tag for tag in predicted_tags if tag is not None]
cm = ConfusionMatrix(gold_tags, predicted_tags)
print(cm.pretty_format(sort_by_count=True, show_percents=True, truncate=10))
tag_mapping = {
    'NOUN': 'NOUN',
    'VERB': 'VERB',
    'ADJ': 'ADJ',
    'ADV': 'ADV',
}
simplified_gold_tags = [tag_mapping.get(tag, tag) for tag in gold_tags]
simplified_predicted_tags = [tag_mapping.get(tag, tag) for tag in predicted_tags]
simplified_cm = ConfusionMatrix(simplified_gold_tags, simplified_predicted_tags)
print(simplified_cm.pretty_format(sort_by_count=True, show_percents=True, truncate=10))
print("###################### Task 29 ######################")
tagged_sents = brown.tagged_sents(categories='news')
bigram_tagger = nltk.BigramTagger(tagged_sents)
tagging_failures = []
for sentence in brown.sents(categories='news'):
    tagged_sentence = bigram_tagger.tag(sentence)
    for word, tag in tagged_sentence:
        if tag is None:
            tagging_failures.append(tagged_sentence)
            break
for failure in tagging_failures[:5]:
    print(failure)
    print()
print("###################### Task 30 ######################")
frequency_threshold = 5
tagged_sents = brown.tagged_sents(categories='news')
words = [word for sent in tagged_sents for word, _ in sent]
word_freq = nltk.FreqDist(words)
preprocessed_tagged_sents = [[(word if word_freq[word] > frequency_threshold else 'UNK', tag) for word, tag in sent] for
                             sent in tagged_sents]
train_size = int(len(preprocessed_tagged_sents) * 0.8)
train_sents = preprocessed_tagged_sents[:train_size]
test_sents = preprocessed_tagged_sents[train_size:]
bigram_tagger = nltk.BigramTagger(train_sents)
accuracy = bigram_tagger.evaluate(test_sents)
print("Bigram Tagger Accuracy:", accuracy)
unigram_tagger = nltk.UnigramTagger(train_sents)
accuracy = unigram_tagger.evaluate(test_sents)
print("Unigram Tagger Accuracy:", accuracy)
default_tagger = nltk.DefaultTagger('NN')
accuracy = default_tagger.evaluate(test_sents)
print("Default Tagger Accuracy:", accuracy)
print("###################### Task 32 ######################")
help(nltk.tag.brill.demo)
train_size = 1000
initial_rules = 10
iterations = 5
nltk.tag.brill.demo(train_size=train_size, initial_rules=initial_rules, iterations=iterations)
print("###################### Task 33 ######################")
tagged_sentences = [
    [('The', 'DT'), ('cat', 'NN'), ('is', 'VBZ'), ('sleeping', 'VBG')],
    [('I', 'PRP'), ('like', 'VB'), ('to', 'TO'), ('eat', 'VB')],
    [('She', 'PRP'), ('is', 'VBZ'), ('reading', 'VBG'), ('a', 'DT'), ('book', 'NN')],
    [('He', 'PRP'), ('plays', 'VBZ'), ('guitar', 'NN')],
]
pos_tags_dict = defaultdict(lambda: defaultdict(set))
for sentence in tagged_sentences:
    for i in range(len(sentence) - 1):
        word, tag = sentence[i]
        next_tag = sentence[i + 1][1]
        pos_tags_dict[word][tag].add(next_tag)
word = 'is'
tag = 'VBZ'
possible_tags = pos_tags_dict[word][tag]
print(f"Possible tags that can follow '{word}' with tag '{tag}': {possible_tags}")
print("###################### Task 34 ######################")
word_counts = defaultdict(set)
for tagged_sent in brown.tagged_sents():
    for word, tags in tagged_sent:
        num_tags = len(tags.split('-'))
        word_counts[num_tags].add(word.lower())
print("Count of Distinct Tags\tCount of Distinct Words")
print("---------------------------------------------")
for i in range(1, 11):
    count = sum(1 for word_set in word_counts.values() if len(word_set) == i)
    print(f"{i}\t\t\t\t{count}")
word_with_max_tags = max(word_counts[3], key=lambda w: len(set(
    tag for sent in brown.tagged_sents(categories='news') for _, tag in sent if
    any(word.lower() == w for word, _ in sent))))
print("\nSentences for the Word with the Greatest Number of Distinct Tags:")
print("--------------------------------------------------------------")
tagged_sentences = brown.tagged_sents(categories='news')
for tagged_sent in tagged_sentences:
    for word, tag in tagged_sent:
        if word.lower() == word_with_max_tags:
            print(f"{tag}:", ' '.join(word for word, _ in tagged_sent))
print("###################### Task 35 ######################")


def extract_features(sentence, index):
    word = sentence[index][0].lower()
    prev_word = sentence[index - 1][0].lower() if index > 0 else None
    next_word = sentence[index + 1][0].lower() if index < len(sentence) - 1 else None

    features = {
        'word': word,
        'prev_word': prev_word,
        'next_word': next_word
    }
    return features


tagged_sentences = brown.tagged_sents(categories='news')
featuresets = []
for tagged_sent in tagged_sentences:
    untagged_sent = nltk.tag.untag(tagged_sent)
    for i, (word, tag) in enumerate(tagged_sent):
        if word.lower() == 'must':
            featuresets.append((extract_features(untagged_sent, i), tag))
train_set = featuresets[:int(len(featuresets) * 0.8)]
test_set = featuresets[int(len(featuresets) * 0.8):]
classifier = NaiveBayesClassifier.train(train_set)
accuracy_score = accuracy(classifier, test_set)
print("Accuracy:", accuracy_score)
example_contexts = [
    "We must go there.",
    "It must be true.",
    "He must study.",
    "We must help them.",
    "She must be joking."
]
print("\nClassification Results:")
print("-----------------------")
for context in example_contexts:
    tokens = nltk.word_tokenize(context.lower())
    tagged_tokens = nltk.pos_tag(tokens)
    for i, (word, tag) in enumerate(tagged_tokens):
        if word == 'must':
            features = extract_features(tagged_tokens, i)
            predicted_tag = classifier.classify(features)
            print("Context:", context)
            print("Predicted Tag:", predicted_tag)
            print()
print("###################### Task 36 ######################")
train_sents = brown.tagged_sents(categories='news')[:1000]
test_sents = brown.tagged_sents(categories='news')[1000:1200]
regexp_tagger = nltk.RegexpTagger([(r'^\d+$', 'CD'), (r'.*', 'NN')])
unigram_tagger1 = nltk.UnigramTagger(train_sents, backoff=regexp_tagger)
unigram_tagger2 = nltk.UnigramTagger(train_sents, backoff=nltk.DefaultTagger('NN'))
unigram_tagger3 = nltk.UnigramTagger(train_sents, backoff=nltk.DefaultTagger('NNP'))
bigram_tagger1 = nltk.BigramTagger(train_sents, backoff=unigram_tagger1)
bigram_tagger2 = nltk.BigramTagger(train_sents, backoff=unigram_tagger2)
bigram_tagger3 = nltk.BigramTagger(train_sents, backoff=unigram_tagger3)
trigram_tagger1 = nltk.TrigramTagger(train_sents, backoff=bigram_tagger1)
trigram_tagger2 = nltk.TrigramTagger(train_sents, backoff=bigram_tagger2)
trigram_tagger3 = nltk.TrigramTagger(train_sents, backoff=bigram_tagger3)
print("Accuracy of Combined Taggers:")
print("----------------------------")
print("Combination 1:", trigram_tagger1.evaluate(test_sents))
print("Combination 2:", trigram_tagger2.evaluate(test_sents))
print("Combination 3:", trigram_tagger3.evaluate(test_sents))
train_sizes = [100, 500, 1000, 2000, 5000, 10000]
print("\nEffect of Training Corpus Size:")
print("-------------------------------")
for size in train_sizes:
    train_subset = brown.tagged_sents(categories='news')[:size]
    trigram_tagger = nltk.TrigramTagger(train_subset, backoff=bigram_tagger1)
    accuracy = trigram_tagger.evaluate(test_sents)
    print(f"Training Size: {size}\tAccuracy: {accuracy}")
print("###################### Task 37 ######################")


class PreviousTagUnigramTagger(UnigramTagger):
    def __init__(self, train, backoff=None):
        super().__init__(train, backoff=backoff)

    def context(self, tokens, index, history):
        if index == 0:
            return None
        else:
            return history[index - 1][1]


train_sents = nltk.corpus.brown.tagged_sents()[:50000]
test_sents = nltk.corpus.brown.tagged_sents()[50000:]
unigram_tagger = PreviousTagUnigramTagger(train_sents)
bigram_tagger = nltk.BigramTagger(train_sents, backoff=unigram_tagger)
trigram_tagger = nltk.TrigramTagger(train_sents, backoff=bigram_tagger)
default_tagger = DefaultTagger('NN')
tagger = nltk.RegexpTagger([(r'^\d+$', 'CD')], backoff=trigram_tagger)
tagger = nltk.BigramTagger(train_sents, backoff=tagger)
tagger = nltk.UnigramTagger(train_sents, backoff=tagger)
tagger = PreviousTagUnigramTagger(train_sents, backoff=tagger)
tagger = nltk.DefaultTagger('NN', backoff=tagger)
accuracy = tagger.evaluate(test_sents)
print("Accuracy:", accuracy)
print("###################### Task 39 ######################")


class LidstoneUnigramTagger(UnigramTagger):
    def context_to_tag(self, tokens, index, history):
        token = tokens[index]
        context = self._context(tokens, index, history)
        return self._unigrams[context].max() if (
                context in self._unigrams and self._unigrams[context].prob(token) > 0) else None


train_sents = nltk.corpus.brown.tagged_sents(categories='news')
unigram_tagger = LidstoneUnigramTagger(train_sents, backoff=nltk.DefaultTagger('NN'), cutoff=1, lambda_=0.7)
bigram_tagger = BigramTagger(train_sents, backoff=unigram_tagger, cutoff=2)
trigram_tagger = TrigramTagger(train_sents, backoff=bigram_tagger, cutoff=3)
test_sents = nltk.corpus.brown.tagged_sents(categories='editorial')
accuracy = trigram_tagger.evaluate(test_sents)
print("Accuracy:", accuracy)
print("###################### Task 40 ######################")
train_data = nltk.corpus.brown.tagged_sents(categories='news')
templates = [
    SymmetricProximateTokensTemplate(brill.ProximateTagsRule, (1, 1)),
    SymmetricProximateTokensTemplate(brill.ProximateTagsRule, (2, 2)),
    SymmetricProximateTokensTemplate(brill.ProximateTagsRule, (1, 2)),
    SymmetricProximateTokensTemplate(brill.ProximateTagsRule, (1, 3)),
    ProximateTokensTemplate(brill.ProximateTagsRule, (-1, -1), (1, 1)),
]
trainer = BrillTaggerTrainer(initial_tagger=nltk.DefaultTagger('NN'))
brill_tagger = trainer.train(train_data, templates=templates)
test_data = nltk.corpus.brown.tagged_sents(categories='editorial')
accuracy = brill_tagger.evaluate(test_data)
print("Accuracy:", accuracy)
print("###################### Task 41 ######################")


class AntiNGramTagger(NgramTagger):
    def __init__(self, n, train_sents, anti_ngrams=None, backoff=None):
        super().__init__(n, train_sents, backoff=backoff)
        self.anti_ngrams = anti_ngrams or []

    def ngram_fd(self, tokens, history):
        if history in self.anti_ngrams:
            return nltk.probability.FreqDist()
        return super().ngram_fd(tokens, history)


train_sents = nltk.corpus.brown.tagged_sents(categories='news')
anti_ngrams = [("the", "the")]
backoff = DefaultTagger('NN')
unigram_tagger = UnigramTagger(train_sents, backoff=backoff)
bigram_tagger = AntiNGramTagger(2, train_sents, anti_ngrams, backoff=unigram_tagger)
trigram_tagger = AntiNGramTagger(3, train_sents, anti_ngrams, backoff=bigram_tagger)

# Test the tagger
test_sents = nltk.corpus.brown.tagged_sents(categories='editorial')
accuracy = trigram_tagger.evaluate(test_sents)
print("Accuracy:", accuracy)
print("###################### Task 42 ######################")


def train_and_evaluate_tagger(tagger_class, train_sents, test_sents):
    tagger = tagger_class(train_sents)
    accuracy = tagger.evaluate(test_sents)
    return accuracy


def cross_validation(tagged_sents, folds, tagger_class):
    fold_size = int(len(tagged_sents) / folds)
    accuracies = []

    for i in range(folds):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size
        test_sents = tagged_sents[test_start:test_end]
        train_sents = tagged_sents[:test_start] + tagged_sents[test_end:]
        accuracy = train_and_evaluate_tagger(tagger_class, train_sents, test_sents)
        accuracies.append(accuracy)

    avg_accuracy = sum(accuracies) / len(accuracies)
    return avg_accuracy


corpus = nltk.corpus.brown
genre_splits = [(corpus.categories(), genre) for genre in corpus.categories()]
source_splits = [(corpus.fileids(), fileid) for fileid in corpus.fileids()]
sentence_splits = [(corpus.sents(fileid), fileid) for fileid in corpus.fileids()]
tagger_class = nltk.tag.UnigramTagger
folds = 5
genre_accuracy = cross_validation(corpus.tagged_sents(), folds, tagger_class)
source_accuracy = cross_validation(corpus.tagged_sents(), folds, tagger_class)
sentence_accuracy = cross_validation(corpus.tagged_sents(), folds, tagger_class)
print("Accuracy based on genre:", genre_accuracy)
print("Accuracy based on source:", source_accuracy)
print("Accuracy based on sentence:", sentence_accuracy)
print("###################### Task 43 ######################")


class MyNgramTagger(nltk.tag.NgramTagger):
    def __init__(self, n, train_sents, backoff=None, **kwargs):
        self._collapsed_vocab = self.collapse_vocab(train_sents)
        super().__init__(n, train_sents, backoff=backoff, **kwargs)

    def collapse_vocab(self, tagged_sents):
        vocab = set()
        for sent in tagged_sents:
            for word, _ in sent:
                vocab.add(word)
        return vocab

    def tag(self, tokens):
        if self._collapsed_vocab is None:
            raise ValueError("The vocabulary has not been collapsed. Please train the tagger first.")

        tokens = [self._lexicalize(token) for token in tokens]
        return super().tag(tokens)

    def _lexicalize(self, tokens):
        return [token if token in self._collapsed_vocab else None for token in tokens]


corpus = nltk.corpus.brown
train_sents = corpus.tagged_sents(categories='news')
test_sents = corpus.tagged_sents(categories='fiction')
n = 3
backoff = nltk.tag.UnigramTagger(train_sents)
tagger = MyNgramTagger(n, train_sents, backoff=backoff)
sent = ["This", "is", "an", "example", "sentence"]
tagged_sent = tagger.tag(sent)
print(tagged_sent)
