import nltk
import re
import time
start = time.time()


# # # Import from NLTK
wn = nltk.corpus.wordnet
stopwords = set(nltk.corpus.stopwords.words('english'))
lancaster = nltk.LancasterStemmer()
brown_tagged = nltk.corpus.brown.tagged_words(tagset='universal')
brown_words = nltk.corpus.brown.words()

gutenberg_words = []
for fileid in nltk.corpus.gutenberg.fileids():
    gutenberg_words = gutenberg_words + list(nltk.corpus.gutenberg.words(fileid))
gutenberg_tagged = nltk.pos_tag(gutenberg_words, tagset='universal')

tagged = list(brown_tagged) + gutenberg_tagged
words = list(brown_words) + gutenberg_words


def filter(a,b):
    # Auxiliary Filter # only those that contains stopwords, o special term/name (big letter)
    # or contains characters other than alphabet will be filtered out
    return (not a in stopwords and not b in stopwords and a.isalpha() and b.isalpha() and
            a.islower() and b.islower() and lancaster.stem(a)!=lancaster.stem(b))


# # # PART 1) FIND HYPHENATED-WORD, Specific D
def prep_fromsynset():
    hyphen = [w for synset in wn.all_synsets() for w in synset.lemma_names()
            if re.search('^[a-z]{3,}-[a-z]+$', w)]
    catch = []  # Catch is list of (word,synset) pair
    for word in hyphen:
        w1, w2 = word.split('-')
        if w1 in stopwords or w2 in stopwords:  # Throw out Stopwords
            continue
        if lancaster.stem(w1)==lancaster.stem(w2):  # Throw out dummy hyphenated words
            continue  # or type_word not in possible
        if similarity(w1,w2):
            catch.append((w1,w2))  # 'w1-w2'
            used_lex_D.add(w1)
    return catch


# # # PART2) Search from the Text and Match with certain Pattern
def search_text():
    catch = []; k=0
    possible = [['ADV', 'ADJ'], ['ADJ', 'NOUN']]  # possible form of D and E, when taken from the text
    while k < len(tagged) - 4:  # -4, to prevent out of index on 'as-adj-as-noun' operations

        if words[k] == 'as' and words[k+2] == 'as':  # SEARCH 'AS-ADJ-AS-NOUN'
            if tagged[k+1][1] == 'ADJ' and tagged[k+3][1] == 'NOUN':
                if filter(words[k+1],words[k+3]):
                    catch.append( (words[k+3], words[k+1], 'AS-ADJ-AS-NOUN') )
            k = k + 4
            continue
            # e.g., as clear as crystal -> crystal clear

        word = words[k]
        if re.search('^[a-z]{3,}-[a-z]+$', word):  # if word contains hyphen
            w1, w2 = word.split('-')
            if not filter(w1,w2):
                k = k+1
                continue
            if tagged[k][1] in ['NOUN','ADJ'] and tagged[k-1][1] == 'DET' and tagged[k+1][1] in ['ADJ','NOUN']:
                catch.append((word,'HYPHEN') )
                k = k + 2
                continue

        word_after = words[k + 1]
        if [tagged[k][1],tagged[k+1][1]] in possible and filter(word, word_after):
            if tagged[k-2][1] in ['ADP','P'] and tagged[k-1][1] == 'DET' and tagged[k+2][1] == 'NOUN':
                catch.append( (words[k] , words[k+1], 'PATTERN') )
                k = k + 3
                continue

        k = k+1
    return catch


def match(catch_synset, catch_text):  # PART3) match catch from text with synset and rank them
    catch = [[],[],[],[],[],[]]
    for tupling in catch_text:

        if tupling[-1] == 'HYPHEN':
            word = tupling[0]
            w1, w2 = word.split('-')
            if (w1, w2) in catch_synset and (w1,w2) not in catch[2]:  # 3RD PRIORITY
                catch[2].append((w1, w2))
            else:
                if w1 in used_lex_D and (w1, w2) not in catch[4] and similarity(a,b):  # Fifth Priority
                    catch[4].append((w1, w2))
        else:
            a, b, c = tupling
            if c == 'AS-ADJ-AS-NOUN':
                if (a,b) in catch_synset and (a, b) not in catch[0]:  # FIRST PRIORITY
                    catch[0].append((a, b))
                else:
                    if a in used_lex_D and (a, b) not in catch[1] : # 2ND Priority and similarity(a,b)
                        catch[1].append((a,b))
            else:  # c == 'PATTERN'
                if (a,b) in catch_synset and (a, b) not in catch[3]: # 4TH Priority
                    catch[3].append((a, b))
                else:
                    if a in used_lex_D and (a, b) not in catch[5] and similarity(a,b): # 6th Priority
                        catch[5].append((a,b))
    result = []
    for w in catch:
        result = result + w + [('BORDERLINE','BETWEEN','PRIORITY')]

    return result
    # those that are found by adj

def similarity(w1,w2): # Make sure D and E are restrictive, for words that are not hyphenated
    if len(wn.synsets(w1)) == 0 or len(wn.synsets(w2)) == 0:
        return False
    index_sim = wn.synsets(w1)[0].path_similarity(wn.synsets(w2)[0])
    return index_sim is not None and 0.25 >= index_sim >= 0.05


# # # MAIN Code

used_lex_D = set()

catch_synset = prep_fromsynset()
catch_text = search_text()
catch = match(catch_synset, catch_text)
catch = catch[:102] # there is borderline to better explain the report

# # # Write CSV Files
import csv
import pathlib

with open(str(pathlib.Path().absolute()) + '\CS372_HW2_output_20180848.csv', 'w', newline='') as f:
    thewriter = csv.writer(f)
    thewriter.writerow(['Intensifier','Word'])
    for index in range(len(catch)):
        thewriter.writerow(catch[index])
# print("--------------END OF PART 4-----------------")


end = time.time();
process_time = end-start
print('Program ends in', process_time)