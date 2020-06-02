import nltk
import time
start = time.time()

# # # PART 1), Accessing the Corpora, we use English book in Gutenberg corpora. Convert to Bigrams
gutenberg = nltk.corpus.gutenberg;   id = 'austen-sense.txt';
text_shake = nltk.Text(gutenberg.words(id));
print(text_shake);
ordered_text = [w.lower() for w in text_shake]  # including punctuations
bigrams_text = list(nltk.bigrams(ordered_text))

# print("--------------END OF PART 1-----------------")


# # # PART 2), Collecting Intensify-Modifying Adverb

# The followings are derived from an HTML website,  Section 'Degree adverbs
url = 'https://www.grammar-quizzes.com/adv_degree.html'
from bs4 import BeautifulSoup
from urllib import request
html = request.urlopen(url).read().decode('utf8')
raw = BeautifulSoup(html, 'lxml').get_text()
tokens = nltk.word_tokenize(raw)

pre_adverb = set()  # adverb comes before the word, such as crowded, almost full
post_adverb = set()  # adverb comes after the word, such as extol, praise highly

# Searching for ly suffix, ly suffix can be both pre and post adverb
ly_suffix = set(w for w in tokens[1271:1369] if w.endswith('ly'))
ly_suffix.update(['insanely','uncommonly,' 'unusually'])  # added manually
# negative words should be handled separately, it will be explained on PART C)
ly_suffix.difference_update(['hardly','terribly', 'slight', 'dreadfully'])

pre_adverb = pre_adverb | ly_suffix; post_adverb = post_adverb | ly_suffix; del ly_suffix;
pre_adverb.update(['pretty','very','well', 'too'])  # added manually
# index 1271:1369 was gained from the tokens.index, it should use hsss parser,
# however, it took too long and there are lots of deprecates

# Searching for without -ly suffix, from the same site. They are mostly considered as pre adverb
without_ly = set(w for w in tokens[1420:1462] if w.isalpha())
without_ly.difference_update({'almost','about','enough','however', 'indeed','rather','quite','right'})
# almost, about, enough, rather, quite, right are ambiguous. They do not always strengthen the intensity.
# however/indeed are more like comparative adverbs.

without_ly.difference_update({'not', 'less', 'least', 'downright', 'dab'})
# negative words should be handled separately, it will be explained on PART C)
pre_adverb = pre_adverb | without_ly; del without_ly

adverb = pre_adverb | post_adverb
# print("--------------END OF PART 2-----------------")



# # # PART 3), Searching for bigrams, that results in the form. Match with available text corpora.
# # # Output should be a list of triples, catch[]

catch = []
import re

def stem(word):
    arr = re.findall(r'^(.*)(ing|ly|ed|ious|ies|ive|es|s|ment|ual)$', word)
    if len(arr) > 0:
        a,b = arr[0];
        return a
    return word
# Finding Word Stem, this is used to increase the lexical diversity of the output, less repetition
# It is also used to describe the similarity of the words that are compared through wordnet


# Access word after and before adverbs

def find_adverb (a,b):
    m = 0; global used_vocabs;
    for synset in wn.synsets(a):
        if '.v.'in str(synset) or '.n.' in str(synset):
            continue  # verb is ignored in my problem, due to its randomness, need bigger analysis
        for lemma in synset.lemmas():
            syn = lemma.name().lower()
            if not syn.isalpha() or syn.endswith('er') or syn.endswith('ly'):  # avoid out-of context similarity
                break;  # avoid numbers such as forty-40, hyphenated words such as bang-up. Avoid adverb and comparative
            if stem(syn) in used_vocabs or syn in filter:
                continue  # Avoid using previously-used words
            if stem(syn) != stem(a): # avoid cases such as advance-advanced
                catch.append([lemma.name().lower(), a, b])
                used_vocabs.add(stem(a)); used_vocabs.add(stem(syn));
                m = 1;
                break;
        if m == 1:
            # trivial code to avoid repetition of pairs/words
            break;


wn = nltk.corpus.wordnet;
stopwords = set(nltk.corpus.stopwords.words('english'))  # stopwords use for further filtration
used_vocabs = set();  # to add lexical variety of the 50 lists of output / avoid repetition
# To produce a more accurate output, lemmas should not be ambiguous, Note that the word 'break' consists of 75 meanings
# according to https://muse.dillfrog.com/lists/ambiguous. The words below are taken
ambiguous = {'break','cut','run','play','make','light','clear','draw','give','hold','set','fall','take','head',\
                  'pass','call','carry','charge','point','catch','check','turn','close','get','right','cover','lift',\
                  'line','open','go','might','about','may','almost','high','narrow','however'}
#  'however' is a comparative verb that can be used in the middle of the sentence without punctuation in literary
#  gutenberg text. It cannot be intensified, so should be removed through the filter
filter = ambiguous | adverb | stopwords;  # filter will exclude lemmas that are ambiguous, adverbial, and common

for a,b in bigrams_text:
    if a in pre_adverb and b.isalpha() and stem(b) not in used_vocabs and b not in filter \
            and not b.endswith('er') and not b.endswith('ly'):  # er/ly suffix are avoided
        find_adverb(b,a)
    elif b in post_adverb and a.isalpha() and stem(a) not in used_vocabs and a not in filter \
            and not a.endswith('er') and not a.endswith('ly'):
        find_adverb(a,b)
# print("--------------END OF PART 3-----------------")


# # # Part 4), Write CSV Files
import csv
import pathlib

with open(str(pathlib.Path().absolute()) + '\CS372_HW1_output_20180848.csv', 'w', newline='') as f:
    thewriter = csv.writer(f)

    thewriter.writerow(['lemma.name()','adverb/word','adverb/word'])
    for index in range(50):
        thewriter.writerow(catch[index])

# print("--------------END OF PART 4-----------------")

end = time.time();
process_time = end-start
print('Program ends in', process_time)