import nltk
import time  # For estimate time complexity
import math # For score system
import csv  # For output
# import os.path # For storing data, which take much time
# import pickle # For storing data, which take much time at birth
import random # For diverse phrase
from nltk.corpus import brown as corpus  # Try various corpus in this line
from nltk.corpus import wordnet as wn

"""
It includes some development environment, code pieces for debugging or convenience
This is not released form.
I would left this to let the instructors know my intention well...!
"""

# Global variables
CORPUS = 'brown'
NOUN = 'n'
VERB = 'v'
ADJ = 'a'
ADJ_SAT = 's'
ADVERB = 'r'
myWordSet = []
THRESHOLD = 0

# Class definition
class Word:
    def __init__(self, name):
        self.name = name                        # word itself
        self.len = len(name)                    # word length
        self.meanings = len(wn.synsets(name))   # How many definitions?
        self.used = 0                           # How many times is it used in text
        self.used_with = 0                      # How many times is It used with adverb
        self.used_as = set()                    # How is it used with adverbs?
        self.syns = []                          # representative synset of this word
        self.representative_synset()

    def __str__(self):
        return self.name + " : " + str(self.meanings) + " meanings, " + str(self.used) + " times used, " + str(self.used_with) + " times used with adverbs " + "as " + str(self.used_as)


    def representative_synset(self, pos = 'n'): # Pick one representative synonyms
        for synset in wn.synsets(self.name):
            title = synset.name()
            cnt = 0
            head = ""
            num_index = -1
            for i in range(len(title)):
                if title[i] == '.' and cnt == 0:
                    head = title[:i]
                    cnt+=1
                elif title[i] == '.' and cnt == 1:
                    num_index = i
            num = int(title[num_index+1:])
            if synset.pos() != pos and head == self.name and len(synset.lemma_names()) > 1 and num <= 10:
                self.syns += synset.lemma_names()

        self.syns = list(set(self.syns))
        self.syns[:] = [syns for syns in self.syns if self.name not in syns or self.name == syns]

# Debug functions
def print_syn(word, opt = 'n'):
    for synset in wn.synsets(word):
        if synset.pos() != opt:
            print(synset.lemma_names())

def print_synsets(word):
    print(wn.synsets(word))

def print_repsyn(word):
    entry = Word(word)
    print(entry.syns)

# Code
'''
Find similar words with the given list of words
input = words : List[String]
        depth : Int - recursive depth
        pos : Char
output = result : Set{String}
'''
def similar_words_multiple(words, depth=5, pos=ADVERB):
    result = set()
    for word in words:
        result.update(similar_words(word, depth, pos))
    return result

'''
Find similar words with the given word in recursive way.
input = word : String
        depth : Int - recursive depth
        pos : Char
output = words : Set{String}
Use wn.synsets for finding synonyms
'''
def similar_words(word, depth=5, pos=ADVERB):
    intermediate = set()  # words in the intermediate recursion
    words = set()  # final output

    # Harvest the similar words, pos has to be same
    synsets = wn.synsets(word)
    for synset in synsets:
        if synset.pos() == pos:
            for name in synset.lemma_names():
                if check_word(name, pos):
                    intermediate.add(name)

    # Recursive step
    if depth != 0:
        for word in intermediate:
            words.update(similar_words(word, depth - 1, pos))
    else:
        words.update(intermediate)

    return words

'''
Helper function for similar_words
input = name : String
        pos : Char - describes the pos of word (NOUN, VERB, ...)
output : bool - whether it is valid, one word synonym
'''
def check_word(name, pos):
    if '_' in name:
        return False
    if pos == ADVERB:
        return check_adverb(name)
    else:
        return True

'''
Check whether the input can be purely recognized by adverb
input = name : String
output : bool
**Currently it just check whether the word ends with 'ing'
'''
def check_adverb(name):
    if name[-3:] == "ing":
        return False
    else:
        return True

'''
Check whether the given word is valid for our target.
It has to be NOT pure noun, it has to be NOT stopwords,
it has to be alphabetical.
'''
def valid_word(word):
    pos_set = set()
    for synset in wn.synsets(word):
        pos_set.add(synset.pos())
    if pos_set == {'n'}:
        return False
    if word.lower() in nltk.corpus.stopwords.words('english'):
        return False
    if word.isalpha():
        return True
    return False

'''
Update word into myWordSet
1. Check the validity of word
2. Check whether there is already a word
3. update attribute
input = word : String
        adverb : String
output = None : Just update myWordSet
'''
def update_word(word, adverb = ""):
    if not valid_word(word):
        return None
    entry = find_word(word)
    if entry != None: # word is already in myWordSet
        entry.used_with += 1
        entry.used_as.add(adverb + " " + word)
        return None
    # New word update
    entry = Word(word)
    entry.used_with += 1
    entry.used_as.add(adverb + " " + word)
    myWordSet.append(entry)
    return None

'''
Update myWordSet
For given input, find phrases including that input, in corpus
For that phrases, update Word attributes.
input = adverbs : List[String]
output = WordSet : List[Words]
'''
def update_wordset(adverbs):
    for fileid in corpus.fileids():
        source = corpus.words(fileid)
        for i in range(len(source)):
            if source[i] in adverbs: # If we find the adverb, update the neightbor words
                cand = source[i+1]
                update_word(cand, source[i])

'''
Return entry if word is in myWordSet
'''
def find_word(word):
    for entry in myWordSet:
        if entry.name == word:
            return entry
    return None

'''
Enlarge the wordset to include the synsets in current myWordSet
It has to be in corpus
input = None
output = None
'''
def enlarge_wordset():
    index = 0
    whole_entry = set()
    for entry in myWordSet:
        for syn in entry.syns:
            whole_entry.add(syn)

    length = len(corpus.fileids())
    for fileid in corpus.fileids():
        print("{} file done for {} files".format(index, length))
        index += 1
        source = corpus.words(fileid) # For each words
        for i in range(len(source)):
            word = source[i]
            for syn in whole_entry:
                if syn == word:
                    found = find_word(syn)
                    if found == None:
                        found = Word(syn)
                        found.used += 1
                        myWordSet.append(found)
                    else:
                        found.used += 1
    print("{} file done for {} files".format(index, length))

'''
Simplify myWordSet
Discard the entry with no syns
Discard the entry with no usage
input = None
output = None
'''
def simplify_wordset():
    myWordSet[:] = [w for w in myWordSet if len(w.syns) != 0]

'''
Delete the phrase or not alphabetical entries
'''
def pure_alpha_wordset():
    myWordSet[:] = [w for w in myWordSet if w.name.isalpha()]

'''
For given Word, calculate the intensity score
1) Compare the length
2) Compare the frequency
3) Compare the number of definitions
input = weak : Word
        strong : Word
output = Tuple(Int, Int, Int)
'''
def get_score(weak, strong):
    if strong == None:
        return -100
    length_score = set_range(get_length_score(weak.len, strong.len), 25)
    freq_score = set_range(get_freq_score(weak, strong), 40)
    def_score = set_range(get_def_score(weak.meanings, strong.meanings), 35)
    return length_score, freq_score, def_score

def get_length_score(weak, strong):
    diff = abs(strong - weak)
    score =  round(diff ** (2.0/3.0) * 4, 2) # 4 point per different char ^ (2/3)
    if strong > weak:
        return score
    else:
        return -score

def get_freq_score(weak, strong):
    diff1 = abs(strong.used - weak.used)
    diff2 = abs(strong.used_with - weak.used_with)
    if diff1 == 0:
        return 0
    score = math.log(diff1, 3) * 3
    if diff2 != 0:
        score += math.log(diff2, 3) * 3
    score = round(score, 2)
    if weak.used > strong.used:
        return score
    else:
        return -score

def get_def_score(weak, strong):
    return (weak - strong) * 1.4

def set_range(score, thr):
    if score > thr:
        return thr
    elif score < -thr:
        return -thr
    else:
        return score

'''
Main function
# means comments or explanation of code
#! means piece of working code, for debugging or convenience
Long comments are the form of String Comment (Just like this line)
'''
# Main function
def main():
    # Get initial set of intensity-modifying adverbs
    # From adverb seeds, find synonyms in recursive way
    adverb_seeds = ['very', 'highly', ]
    print("It's extracting intensity modifying adverbs...")
    start_tick = time.time()
    adverb_set = similar_words_multiple(adverb_seeds, 3, ADVERB)
    print("The result is : {}".format(adverb_set))
    print("Synonym extraction processing time : {}".format(time.time() - start_tick))

    #! Convenience
    #! if os.path.isfile('{}.txt'.format(CORPUS)): # myWordSet for CORPUS exists
    #!    print("We use existing WordSet : Loading...")
    #!    start_tick = time.time()
    #!    f = open('{}.txt'.format(CORPUS), 'rb')
    #!    myWordSet[:] = pickle.load(f)
    #!    f.close()
    #!    print("Loading myWordSet processing time : {}".format(time.time() - start_tick))

    #! else:
    # Previous myWordSet doesn't exist, create new one and store it in file
    # Investigate corpus and find the combination of phrase "adverb + word"
    # Store the word in myWordSet, update attributes
    print("Creating new WordSet")
    start_tick = time.time()
    update_wordset(adverb_set)
    print("Building First myWordSet processing time : {}".format(time.time() - start_tick))

    # Simplify the wordset, deleting entry with no synonyms
    start_tick = time.time()
    simplify_wordset()
    print("Simplifying myWordSet processing time : {}".format(time.time() - start_tick))

    # Now the initial resources are done
    # Count the whole words and synonyms for myWordSet in corpus
    # If the word exists, add it to myWordSet
    start_tick = time.time()
    enlarge_wordset()
    pure_alpha_wordset()
    print("Enlarging myWordSet processing time : {}".format(time.time() - start_tick))

    #! Convenience
    #! This is for saving myWordSet
    #! f = open('{}.txt'.format(CORPUS), 'wb')
    #! pickle.dump(myWordSet[:], f)
    #! f.close()

    print("myWordSet entries : {}".format(len(myWordSet)))

    '''
    Now our wordset is ready to analyze
    We will make score from -100 to 100 to determine the intensity between two words
    
    We promise to say,
    'Strong words' are higher at intensity than 'weak words'.
    
    Three hypotheses for score:
    1) Strong words usually longer than weak words
    2) Strong words usually less used than weak words
    3) Strong words usually have less definitions than weak words
    
    We would score each synonyms for entry.
    For example, let's say 'great' has synonyms ['good', 'well', 'great', 'excellent', 'outstanding']
    Then, each score would be                   [-30,    -40,    0,       70,          80           ]
    
    And then extract the pair in the form of list, [very great, excellent]
    write it in csv
    '''
    start_tick = time.time()
    avg_len_score = 0
    avg_freq_score = 0
    avg_def_score = 0
    cnt_output = 0
    output = []
    for weak in myWordSet:
        if weak.used_as == set(): continue  # It is not used with adverb
        for syn in weak.syns:
            strong = find_word(syn) # Check the syn is used in corpora
            if strong != None:
                len_score, freq_score, def_score = get_score(weak, strong)
                if len_score + freq_score + def_score > THRESHOLD:
                    # Calculate the avg score for only passed the condition
                    # insert each scores for debugging
                    avg_len_score += len_score
                    avg_freq_score += freq_score
                    avg_def_score += def_score
                    cnt_output += 1
                    phrase = random.choice(list(weak.used_as))
                    output_tuple = (phrase, syn, len_score + freq_score + def_score, len_score, freq_score, def_score)
                    output.append(output_tuple)
    output.sort(key = lambda element: element[2])
    output.reverse()
    print("Score processing time : {}".format(time.time() - start_tick))
    #! print("Average length score : {}".format(avg_len_score/cnt_output))
    #! print("Average freq score : {}".format(avg_freq_score/cnt_output))
    #! print("Average def score : {}".format(avg_def_score/cnt_output))
    #! print("Number of candidates : {}".format(len(output)))

    # Open the csv file, and make writer
    f = open('CS372_HW1_output_20170490.csv', 'w', encoding='utf-8', newline='')
    csvwriter = csv.writer(f)
    cnt = 0
    for i in output:
        csvwriter.writerow(list(i)[:2])
        cnt += 1
        if cnt >= 50:
            break
    print("Output is generated")

    # Write and Close the csv file
    f.close()

main()