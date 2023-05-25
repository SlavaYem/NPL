import nltk
import re
from nltk.stem import PorterStemmer, LancasterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import abc
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import cmudict
from nltk.corpus import wordnet as wn
from nltk.text import Text
from nltk.corpus import udhr
from nltk import FreqDist
from nltk.metrics import spearman_correlation
from nltk.corpus import wordnet

print("###################### Task 1 ######################")
text = "I love playing football and I enjoy playing with my friends."
tokens = word_tokenize(text)
porter_stemmer = PorterStemmer()
lancaster_stemmer = LancasterStemmer()
porter_stemmed_words = [porter_stemmer.stem(word) for word in tokens]
lancaster_stemmed_words = [lancaster_stemmer.stem(word) for word in tokens]
print("Porter Stemmer:")
for word in porter_stemmed_words:
    print(word)

print("\nLancaster Stemmer:")
for word in lancaster_stemmed_words:
    print(word)
print("###################### Task 2 ######################")
silly = "I am feel quite silly today and enjoying the sunshine."
try:
    index_of_e = silly.index('e')
    index_of_re = silly.index('re')
    print("Index of 'e':", index_of_e)
    print("Index of 're':", index_of_re)
except ValueError:
    print("Substring not found")
words = ['apple', 'banana', 'cherry', 'date', 'elderberry']
try:
    word_position = words.index('cherry')
    print("Position of 'cherry':", word_position)
except ValueError:
    print("Word not found in the list")
try:
    index_of_in = silly.index('in')
    phrase = silly[:index_of_in].split()
    print("Phrase:", phrase)
except ValueError:
    print("'in' not found in the string")
print("###################### Task 3 ######################")


def convert_nationality_to_noun(adjective):
    nationality_mapping = {
        'Canadian': 'Canada',
        'Australian': 'Australia',
    }

    return nationality_mapping.get(adjective, adjective)


adjective = 'Canadian'
noun = convert_nationality_to_noun(adjective)
print(noun)
adjective = 'Australian'
noun = convert_nationality_to_noun(adjective)
print(noun)
adjective = 'French'
noun = convert_nationality_to_noun(adjective)
print(noun)
print("###################### Task 4 ######################")
# corpus = nltk.corpus.brown
# tokenized_text = nltk.word_tokenize(corpus.raw())
# pattern = r"as best (?:as )?[A-Za-z]+ can"
# phrases = nltk.Text(tokenized_text).findall(pattern)
# if phrases:
#     for phrase in phrases:
#         print(phrase)
# else:
#     print("No matches found for the given pattern.")
print("###################### Task 5 ######################")


def convert_to_lolspeak(word):
    replacements = {
        r"\b(?:Hello|Hi)\b": "Ohai",
        r"\b(?:world)\b": "wurld",
        r"\b(?:God|Lord|Heaven)\b": "Ceiling Cat",
    }

    for pattern, replacement in replacements.items():
        word = nltk.re.sub(pattern, replacement, word)
    return word


lolcat_text = nltk.corpus.genesis.words('lolcat.txt')
converted_text = [convert_to_lolspeak(word) for word in lolcat_text]
print(" ".join(converted_text))
print("###################### Task 6 ######################")


def remove_html_tags(html):
    # Remove HTML tags
    cleaned_text = re.sub(r"<.*?>", "", html)
    return cleaned_text


def normalize_whitespace(text):
    # Normalize whitespace
    cleaned_text = re.sub(r"\s+", " ", text)
    return cleaned_text


with open('example.html', 'r') as file:
    html_content = file.read()
cleaned_text = remove_html_tags(html_content)
normalized_text = normalize_whitespace(cleaned_text)
print(normalized_text)
print("###################### Task 7 ######################")
text = "This is a test string with long-\nterm and encyclo-\npedia."
modified_text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
# modified_text = re.sub(r"(\w+)-\n(\w+(?:-|ing|ed|s|es)?)", r"\1\2", text)
print(modified_text)
print("###################### Task 8 ######################")


def soundex(name):
    name = name.upper()
    soundex_mapping = {
        'BFPV': '1', 'CGJKQSXZ': '2', 'DT': '3', 'L': '4',
        'MN': '5', 'R': '6', 'AEIOUHWY': ''}
    first_letter = name[0]
    name = ''.join([char for char in name[1:] if char.isalpha()])
    soundex_code = [first_letter]
    for char in name:
        for key in soundex_mapping:
            if char in key:
                code = soundex_mapping[key]
                if code not in soundex_code:
                    soundex_code.append(code)
    soundex_code = soundex_code[:4] + ['0'] * (4 - len(soundex_code))
    soundex_code = ''.join(soundex_code)
    return soundex_code


name = "Slava"
soundex_code = soundex(name)
print(f"The Soundex code for '{name}' is: {soundex_code}")
print("###################### Task 9 ######################")


def compute_reading_difficulty_score(text):
    tokenizer = PunktSentenceTokenizer()
    sentences = tokenizer.tokenize(text)
    difficult_words_count = 0
    pronouncing_dict = cmudict.dict()
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        for word in words:
            word = word.lower()
            word = word.strip('\'".,?!()[]{}')
            if word in pronouncing_dict:
                phonemes = pronouncing_dict[word][0]
                syllable_count = sum([1 for phoneme in phonemes if phoneme[-1].isdigit()])
                if syllable_count > 2:
                    difficult_words_count += 1
    total_words = len(nltk.word_tokenize(text))
    difficulty_score = difficult_words_count / total_words
    return difficulty_score


rural_news = abc.raw('rural.txt')
science_news = abc.raw('science.txt')
rural_difficulty_score = compute_reading_difficulty_score(rural_news)
science_difficulty_score = compute_reading_difficulty_score(science_news)
print("ABC Rural News Difficulty Score:", rural_difficulty_score)
print("ABC Science News Difficulty Score:", science_difficulty_score)
print("###################### Task 10 ######################")
words = ['attribution', 'confabulation', 'elocution', 'sequoia', 'tenacious', 'unidirectional']
vsequences = sorted({''.join([char for char in word if char in 'aeiou']) for word in words})
print(vsequences)
print("###################### Task 11 ######################")


def create_semantic_index(text):
    indexed_words = {}
    tokens = text.tokens
    for index, word in enumerate(tokens):
        if word.isalpha():
            synsets = wn.synsets(word)
            if synsets:
                word_offset = synsets[0].offset()
                hypernym_offsets = [hypernym.offset() for synset in synsets for hypernym in synset.hypernyms()]
                indexed_words[word] = (word_offset, hypernym_offsets)
    return indexed_words


def concordance_search(text, word):
    concordance_results = []
    tokens = text.tokens
    word_length = len(word)
    for i in range(len(tokens)):
        if tokens[i] == word and i + word_length < len(tokens):
            left_context = " ".join(tokens[i - 5: i])
            right_context = " ".join(tokens[i + word_length: i + word_length + 5])
            concordance_results.append((word, left_context, right_context))

    return concordance_results


text_collection = nltk.Text(nltk.corpus.gutenberg.words('shakespeare-macbeth.txt'))
semantic_index = create_semantic_index(text_collection)
concordance_results = concordance_search(text_collection, 'blood')
processed_words = set()
for concordance in concordance_results:
    word = concordance[0]
    if word not in processed_words:
        processed_words.add(word)
        if word in semantic_index:
            word_offset, hypernym_offsets = semantic_index[word]
            print(f"Word: {word}")
            print(f"Synset Offset: {word_offset}")
            print(f"Hypernym Offsets: {hypernym_offsets}")
print("###################### Task 12 ######################")
# languages = ['English', 'Spanish', 'French']
# nltk.download('udhr')
# texts = [udhr.raw(lang) for lang in languages]
# tokenized_texts = [list(text) for text in texts]
# freq_distributions = [FreqDist(text) for text in tokenized_texts]
# results = []
# for i in range(len(languages)):
#     for j in range(i + 1, len(languages)):
#         correlation = spearman_correlation(freq_distributions[i], freq_distributions[j])
#         results.append((languages[i], languages[j], correlation))
# results.sort(key=lambda x: x[2], reverse=True)
# for lang1, lang2, correlation in results:
#     print(f"Correlation between {lang1} and {lang2}: {correlation:.4f}")
print("###################### Task 13 ######################")


def find_novel_senses(text):
    words = nltk.word_tokenize(text)
    novel_senses = set()
    for word in words:
        synsets = wordnet.synsets(word)
        context = [words[i] for i in range(len(words)) if words[i] != word and abs(i - words.index(word)) <= 2]
        for synset in synsets:
            similarity_scores = [synset.path_similarity(wordnet.synsets(context_word)[0]) for context_word in context if
                                 wordnet.synsets(context_word)]
            if any(score and score > 0.2 for score in similarity_scores):
                novel_senses.add(word)
    return novel_senses


text = "The bank is located near the river bank."
novel_words = find_novel_senses(text)
print("Words with novel senses:", novel_words)
print("###################### Task 14 ######################")
normalization_rules = {
    "Pls": "please",
    "Thx": "thanks",
    "msg": "message",
    "ASAP": "as soon as possible"
}


def normalize_text(text):
    words = re.findall(r'\w+|\S+', text)
    normalized_words = []
    for word in words:
        if word in normalization_rules:
            normalized_words.append(normalization_rules[word])
        else:
            normalized_words.append(word)
    normalized_text = ' '.join(normalized_words)
    return normalized_text


text = "please send the msg ASAP. Thx!"
normalized_text = normalize_text(text)
print("Normalized text:", normalized_text)
