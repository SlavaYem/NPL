from nltk import word_tokenize
import nltk
from nltk import FreqDist, Text


def last_n(text, n):
    my_text = word_tokenize(text)
    last_word = sorted(my_text[-n::])
    for elem in last_word:
        print(elem)


text_first = """Python was conceived in the late 1980s[42] by Guido van Rossum at Centrum Wiskunde & Informatica (CWI) in the Netherlands as a successor to the ABC programming language, which was inspired by SETL,[43] capable of exception handling and interfacing with the Amoeba operating system.[13] Its implementation began in December 1989.[44] Van Rossum shouldered sole responsibility for the project, as the lead developer, until 12 July 2018, when he announced his "permanent vacation" from his responsibilities as Python's "benevolent dictator for life", a title the Python community bestowed upon him to reflect his long-term commitment as the project's chief decision-maker.[45] In January 2019, active Python core developers elected a five-member Steering Council to lead the project.[46][47]

Python 2.0 was released on 16 October 2000, with many major new features such as list comprehensions, cycle-detecting garbage collection, reference counting, and Unicode support.[48] Python 3.0, released on 3 December 2008, with many of its major features backported to Python 2.6.x[49] and 2.7.x. Releases of Python 3 include the 2to3 utility, which automates the translation of Python 2 code to Python 3.[50]

Python 2.7's end-of-life was initially set for 2015, then postponed to 2020 out of concern that a large body of existing code could not easily be forward-ported to Python 3.[51][52] No further security patches or other improvements will be released for it.[53][54] Currently only 3.7 and later are supported. In 2021, Python 3.9.2 and 3.8.8 were expedited[55] as all versions of Python (including 2.7[56]) had security issues leading to possible remote code execution[57] and web cache poisoning.[58]"""


last_n(text_first, 20)


def analyze_text(text):
    words = nltk.word_tokenize(text)
    filtered_words = [word for word in words if len(word) >= 5 and word.isalpha()]
    fdist = FreqDist(filtered_words)
    print('Top 3 most common words:')
    for word, frequency in fdist.most_common(3):
        print(f'{word}: {frequency} times')
    if fdist:
        most_common_word = fdist.most_common(1)[0][0]
        text = Text(filtered_words)
        text.dispersion_plot([most_common_word])


analyze_text(text_first)
