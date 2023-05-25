import random
import nltk
import requests
from bs4 import BeautifulSoup
import re
from urllib import request
from nltk.corpus import words
from nltk import ngrams

#
# url = "https://www.gutenberg.org/cache/epub/70833/pg70833.txt"
# response = request.urlopen(url)
# raw = response.read().decode('utf8')
# print(type(raw))
# print(len(raw))
# print(raw[:75])
# proxies = {'http': 'http://www.someproxy.com:3128'}
# print(request.ProxyHandler(proxies))
# tokens = word_tokenize(raw)
# print(type(tokens))
# print(len(tokens))
# print(tokens[:10])
# text = nltk.Text(tokens)
# print(type(text))
# print(text[1024:1062])
# print(text.collocations())
# print(raw.find("PART"))
# print(raw.rfind("End of Project Gutenberg's Crime"))
# raw = raw[5338:1157743]
# print(raw.find("PART I"))
print("###################### Task 1 ######################")
s = 'colorless'
s = s[:4] + 'u' + s[4:]
print(s)
print("###################### Task 2 ######################")
word1 = 'dish-es'[:4]
word2 = 'run-ning'[:3]
word3 = 'nation-ality'[:6]
word4 = 'un-do'[:2]
word5 = 'pre-heat'[:3]
print(word1, word2, word3, word4, word5)
print("###################### Task 4 ######################")
monty = "Monty Python"
result1 = monty[6:11:2]
result2 = monty[10:5:-2]
result3 = monty[::3]
result4 = monty[1:9:4]
result5 = monty[::-1]
print(result1)
print(result2)
print(result3)
print(result4)
print(result5)
print("###################### Task 5 ######################")
result6 = monty[::-1]
print(result6)
print("###################### Task 6 ######################")
patterns = [
    r'[a-zA-Z]+',
    r'[A-Z][a-z]*',
    r'p[aeiou]{,2}t',
    r'\d+(\.\d+)?',
    r'([^aeiou][aeiou][^aeiou])*',
    r'\w+|[^\w\s]+'
]

textes = [
    "Red dog --3.",
    "Red dog Red",
    "patrik patt ",
    "Value = 12.3 99",
    "bab ab ab nen",
    "-. Red ^44 ?"
]

for pattern, text in zip(patterns, textes):
    nltk.re_show(pattern, text)
print("###################### Task 7 ######################")
text = "The a an goood. 2*3+8"
patterns = [
    r'\b(a|an|the|A|An|The)\b',
    r'\d+([*+]\d+)*'
]
for pattens in patterns:
    nltk.re_show(pattens, text)
print("###################### Task 8 ######################")


def remove_html_markup(url):
    response = request.urlopen(url)
    html = response.read().decode('utf8')
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text()
    return text


url = 'http://nltk.org/'
content_without_markup = remove_html_markup(url)
print(content_without_markup)
print("###################### Task 9 ######################")


def save_text_to_file(text, filename):
    with open(filename, 'w') as file:
        file.write(text)


text = "This is an example text."
save_text_to_file(text, 'corpus.txt')


def load(filename):
    with open(filename, 'r') as file:
        text = file.read()
    return text


loaded_text = load('corpus.txt')
print(loaded_text)
text = "This is an example sentence, with some punctuation marks: !?.,;"
pattern = r'''(?x)   # Verbose flag for multiline regular expression
    \w+              # Matches one or more word characters
    |                # OR
    [^\w\s]+         # Matches one or more non-word and non-space characters
'''

tokens = nltk.regexp_tokenize(text, pattern)
print(tokens)

text = "Monetary amount: $100.00, Date: 2023-05-23, Person: John Doe, Organization: OpenAI"
monetary_amount_pattern = r'\$\d+(\.\d+)?'
date_pattern = r'\d{4}-\d{2}-\d{2}'
name_pattern = r'[A-Z][a-z]+ [A-Z][a-z]+'
organization_pattern = r'[A-Z][a-z]+'

patterns = [
    monetary_amount_pattern,
    date_pattern,
    name_pattern,
    organization_pattern
]

tokens = []
for pattern in patterns:
    tokens += nltk.regexp_tokenize(text, pattern)

print(tokens)
print("###################### Task 10 ######################")
url = "https://en.wikipedia.org/wiki/Interrogative_word"
text = remove_html_markup(url)
pattern = r'\bwh\w+\b'
wh = set(nltk.regexp_tokenize(text, pattern))
print(wh)
print("###################### Task 11 ######################")
filename = 'word_frequencies.txt'
lines = open(filename).readlines()
word_freq_pairs = []
for line in lines:
    fields = line.split()
    if len(fields) != 2:
        continue
    word, freq = fields
    try:
        freq = int(freq)
        word_freq_pairs.append([word, freq])
    except ValueError:
        continue
print(word_freq_pairs)
print("###################### Task 12 ######################")


def get_temperature(city):
    base_url = 'https://www.weather-forecast.com'
    search_url = f'{base_url}/locations/{city}/forecasts/latest'
    response = requests.get(search_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        temperature_element = soup.find(class_='temp')
        if temperature_element:
            temperature = temperature_element.get_text().strip()
            return temperature
        else:
            print('Не вдалося знайти елемент з прогнозом температури.')
    else:
        print('Помилка при доступі до веб-сайту з погодою.')


city = 'Kyiv'
temperature = get_temperature(city)
if temperature:
    print(f"Прогнозована максимальна температура в місті {city} сьогодні: {temperature}")

print("###################### Task 13 ######################")


def unknown(url):
    response = request.urlopen(url)
    html = response.read().decode('utf8')
    words_list = re.findall(r'[a-z]+', html.lower())
    words_corpus = set(words.words())
    unknown_words = [word for word in words_list if word not in words_corpus]
    return unknown_words


url = 'http://example.com'
unknown_words = unknown(url)
print(unknown_words)
print("###################### Task 14 ######################")


def unknown(url):
    response = request.urlopen(url)
    html = response.read().decode('utf8')
    html = re.sub(r'<.*?>', ' ', html)
    words_list = re.findall(r'\b[a-z]+\b', html.lower())
    words_corpus = set(words.words())
    unknown_words = [word for word in words_list if word not in words_corpus]
    return unknown_words


url = 'http://news.bbc.co.uk/'
unknown_words = unknown(url)
print(unknown_words)
print("###################### Task 15 ######################")
text = "I don't know what you're talking about."
pattern = r"\b\w+(?:n't|\b)"
tokens = nltk.regexp_tokenize(text, pattern)
print(tokens)
print("###################### Task 16 ######################")
nltk.download('udhr')
text = nltk.corpus.udhr.words('Hungarian_Magyar-Latin1')
front_vowels = ['e', 'é', 'i', 'í', 'ö', 'ő', 'ü', 'ű']
back_vowels = ['a', 'á', 'o', 'ó', 'u', 'ú']


def categorize_vowel(vowel):
    if vowel in front_vowels:
        return 'Front'
    elif vowel in back_vowels:
        return 'Back'
    else:
        return None


vowel_sequences = []
vowel_bigrams = {}

for word in text:
    vowels = [char.lower() for char in word if char.lower() in 'aeiouáéíóöőúüű']
    vowel_sequences.extend(vowels)
    for a, b in ngrams(vowels, 2):
        a_category = categorize_vowel(a)
        b_category = categorize_vowel(b)
        if a_category and b_category:
            bigram_key = (a_category, b_category)
            vowel_bigrams[bigram_key] = vowel_bigrams.get(bigram_key, 0) + 1
print("Vowel Bigram Table:")
for bigram, count in vowel_bigrams.items():
    print(f"{bigram[0]}-{bigram[1]}: {count}")
    
print("###################### Task 17 ######################")
letters = ''.join(random.choice("aehh ") for _ in range(500))
normalized = ' '.join(letters.split())
print(normalized)
print("###################### Task 18 - 19 ######################")


def compute_ari(text):
    words = nltk.corpus.brown.words(categories=text)
    sentences = nltk.corpus.brown.sents(categories=text)
    total_letters = sum(len(word) for word in words)
    total_words = len(words)
    total_sentences = len(sentences)
    average_letters_per_word = total_letters / total_words
    average_words_per_sentence = total_words / total_sentences
    ari_score = 4.71 * average_letters_per_word + 0.5 * average_words_per_sentence - 21.43
    return ari_score


ari_f = compute_ari('lore')
print("ARI score for section f (lore):", ari_f)
ari_j = compute_ari('learned')
print("ARI score for section j (learned):", ari_j)
