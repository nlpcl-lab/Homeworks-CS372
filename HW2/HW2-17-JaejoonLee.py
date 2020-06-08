import nltk
import time  # For estimate time complexity
import math # For score system
import csv  # For output
import os.path # For storing data, which take much time at birth CONVENIENCE
import pickle # For storing data, which take much time at birth CONVENIENCE
import random # For diverse phrase
from nltk.corpus import brown as corpus  # Chose brown corpus to ensure formal uses of words
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer

"""
It includes some development environment, code pieces for debugging or convenience
This is not released form.
I would left this to let the instructors know my intention well...!
"""

#### Debug functions
def print_def(word, token = ' '):
    tab = '    '
    print("Definitions of {}".format(word))
    for synset in wn.synsets(word):
        if token in synset.definition():
            print(tab + "{} : {}".format(synset, synset.definition()))


def word_freq(word):
    cnt = 0
    for a, b in corpus.tagged_words():
        if a == word:
            cnt += 1
    print("{} , {} times used".format(word, cnt))
    return cnt


def valid_word(word):
    pos_set = set()
    for synset in wn.synsets(word):
        pos_set.add(synset.pos())
    if pos_set == {'n'}:
        return False
    if word.lower() in nltk.corpus.stopwords.words('english'):
        return False
    if word.isalpha() and word.islower():
        return True
    return False


def find_bigram(front, back):
    for (a, b) in nltk.bigrams(corpus.tagged_words(tagset = 'universal')):
        if a[0] == front and b[0] == back:
            print (a, b)


def find_bigram_front(front):
    for (a, b) in nltk.bigrams(corpus.tagged_words(tagset = 'universal')):
        if a[0] == front:
            print (a, b)


def find_bigram_back(back):
    for (a, b) in nltk.bigrams(corpus.tagged_words(tagset = 'universal')):
        if b[0] == back:
            print (a, b)


#### Body functions
def word_def_contains(keyword):
    '''
    Extracts words that contains keyword in its definition (not noun def)
    '''
    result = []
    words = [i for i in wn.all_synsets() if i.pos() != 'n' and keyword in i.definition()]
    for i in words:
        result += i.lemma_names()
    return [i for i in list(set(result)) if i.isalpha()]


def filter_defs(adverbs):
    '''
    Filter the list of words containing 'degree' in their defs
    1. Delete which has 'degrees' in its def rather than 'degree' since it has meaning of calibration
    2. Only qualify which has 'to a/an ~ degree' or 'degree or extent' (vice versa)
    return purified adverbs
    '''
    dels = []
    for i in range(len(adverbs)):
        for synset in wn.synsets(adverbs[i]):
            if synset.pos() in ['a', 's', 'r', 'v'] and 'degree' in synset.definition(): # Among the raw chosen
                if 'degrees' in synset.definition():
                    dels.append(adverbs[i])
                elif not meet_condition(synset.definition()):
                    dels.append(adverbs[i])
    return [adverb for adverb in adverbs if adverb not in dels]


def meet_condition(definition):
    '''
    return Bool whether it meets condition 2, in filter_defs()
    Recall) 2. Only qualify which has 'to a/an ~ degree/extent (vice versa)
    '''
    defs = nltk.word_tokenize(definition)
    for i in range(len(defs)-1):
        if defs[i] == 'to' and defs[i+1] in ['a', 'an']:
            if 'degree' in defs[i+1:] or 'extent' in defs[i+1:]:
                return True
    return False


def data_init(data, adverbs):
    '''
    Init dataset
    We would use nested dictionary data structure for entire usage data
    '''
    for adverb in adverbs:
        data[adverb] = {'freq': 0, 'used':{}}


def update_data(data, adverbs):
    '''
    Update dataset searching through the bigrams
    Check which lexical items are used with the intensifiers, in a correct semantics
    To enhance classification, lemmatize each lexical items
    Update usage frequency for each phrases
    '''
    cnt = 0
    corpus_tagged = corpus.tagged_words()
    for (a, b) in nltk.bigrams(corpus_tagged):
        if a[0] in adverbs and check_semantics(a[1], b[1]) and b[0].isalpha() and b[0].islower():
            dict_temp = data[a[0]]
            dict_temp['freq'] += 1
            b = lemmatize(b)
            if b[0] in dict_temp['used']:
                dict_temp['used'][b[0]] += 1
            else:
                dict_temp['used'][b[0]] = 1


def lemmatize(b):
    '''
    Lemmatize lexical items
    For NOUN : Change plural to singular, including some corner cases not handled with WordNetLemmatizer
    For VERB : Change to original, no corner case
    For ADJ : Change to original, no corner case
    For ADV : Change to original, no corner case
    '''
    wnl = WordNetLemmatizer()
    ADJ = ['JJR', 'JJS', 'JJT']
    NOUN = ['NNS'] # Ignore 'NP' 'NPS' 'NR'
    VERB = ['VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    ADV = ['RBR', 'RBT', 'RN', 'RP']
    if b[1] in NOUN:
        lemmatized = wnl.lemmatize(b[0], 'n')
        if lemmatized != b[0]:
            return lemmatized, 'NN'
        else:
            return lemmatize_corner_case(b)
    elif b[1] in VERB:
        lemmatized = wnl.lemmatize(b[0], 'v')
        return lemmatized, 'VB'
    elif b[1] in ADJ:
        lemmatized = wnl.lemmatize(b[0], 'a')
        return lemmatized, 'JJ'
    elif b[1] in ADV:
        lemmatized = wnl.lemmatize(b[0], 'r')
        return lemmatized, 'RB'
    else:
        return b


def lemmatize_corner_case(b):
    '''
    Here are some corner cases that WordNetLemmatizer cannot change plural to singular
    coeds, people, men, beasties, headquarters, clothes
    If the plural can be singular, we would change both item and tag
    If the plural itself has meaning, we would preserve it and just change the tag
    '''
    if b[0] == 'coeds':
        return 'coed', 'NN'
    elif b[0] == 'people':
        return 'person', 'NN'
    elif b[0] == 'men':
        return 'man', 'NN'
    elif b[0] == 'beasties':
        return 'beastie', 'NN'
    else:
        return b[0], 'NN'


def check_semantics(a, b):
    '''
    Check the tags and get only valid semantics
    'NP' 'NPS' are not used
    'NR' 'RP' is not appropriate tags for lexical items
    '''
    ADJ = ['JJ', 'JJR', 'JJS', 'JJT']
    NOUN = ['NN', 'NNS'] # Ignore 'NP' 'NPS' 'NR'
    VERB = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    ADV = ['RB', 'RBR', 'RBT', 'RN'] # Ignore 'RP'
    return ((a in ADJ and b in NOUN)
            or (a in ADV and b in VERB+ADJ+ADV)
            or (a == 'QL' and b in VERB+ADJ+ADV) # qualifier
            or (a == 'AP' and b in NOUN)) # post-determiner + NOUN


def get_commonity_score(adverb, data_dict):
    '''
    This scores out how common the intensifier is.
    Criteria 1) How many definition it has
    Criteria 2) Rank of definition as intensifier
    Criteria 3) How many it is used with lexical item E
    Calibration is
    Cri3 * (Cri2)/Sum(1, ... , Cri1)
    '''
    first_cri = 0
    second_cri = 0
    third_cri = data_dict['freq']
    for synset in wn.synsets(adverb):
        if synset.pos() in ['a', 's', 'r', 'v']:
            first_cri += 1
        if 'intensifier' in synset.definition() or 'degree' in synset.definition():
            second_cri = first_cri
    entity = sum(range(1, first_cri+1))
    part = float(second_cri)/float(entity)
    return round(third_cri * part / 5, 3)


def find_common_score(common_score, word):
    for i in common_score:
        if i[0] == word:
            return i[1]


def init_elements(elements, dataset):
    '''
    Initialize dictionary for lexical item E, elements
    freq : How many it is used in corpus?
    adjective : How many adjective defs?
    used_common : Usage with common intensifiers
    used_uncommon : Usage with uncommon intensifiers
    '''
    for i in dataset:
        for word in dataset[i]['used']:
            elements[word] = {'freq' : 0, 'adjective' : 0, 'used_common' : {}, 'used_uncommon' : {}}


def update_elements(elements, dataset, common_adverbs):
    '''
    Update elements
    freq : See through corpus
    adjective : See through synsets
    used_common/uncommon : See through dataset
    '''
    # freq
    for fileid in corpus.fileids():
        for word in corpus.words(fileid):
            if word in elements:
                elements[word]['freq'] += 1
    # adjective
    for element in elements:
        for synset in wn.synsets(element):
            if synset.pos() in ['a', 's', 'r']:
                elements[element]['adjective'] += 1
    # used_common
    for common in common_adverbs:
        data_dict = dataset[common[0]]['used']
        for element in data_dict:
            num = data_dict[element]
            elements[element]['used_common'][common[0]] = num
    # used_uncommon
    uncommon_adverbs = [word for word in dataset if word not in [common for common, score in common_adverbs]]
    for uncommon in uncommon_adverbs:
        data_dict = dataset[uncommon]['used']
        for element in data_dict:
            num = data_dict[element]
            elements[element]['used_uncommon'][uncommon] = num


def accuracy_score(scoreboard, elements, common_score):
    '''
    Focused on how accurately the adverb is used to change intensity of following.
    LOW SCORE, HIGH ACCURACY
    1. Gather the entity of its usage, for lexical item E, sigma(commonity_of_intensifier * intensifier_freq)
        For example, lexicla item E 'people', used 3 times with 'very' and 4 times with 'dead', freq is 12
        no adjective definition
        entity = commonity(very) * 3 + commonity(dead) * 4
        for 'dead people', the accuracy score is
        (commonity(dead) * 4 / entity) * freq / (adj+1)
    adj+1 is for complementing the case adj == 0
    adj gives an advantage for adjectives that it has potential to be used as adjective, which is scalable.
    2. If it is used with common adverbs, give penalty of +10 score.
    This is restriction score rather than accuracy score, but it was hard to change after first implementation.
    In further step, restriction score would be directly added to accuracy score to generate total score.
    '''
    for e in elements:
        freq = elements[e]['freq'] + 3 # Complementation for freq = 0 due to lemmatization
        adj = elements[e]['adjective']
        com = elements[e]['used_common']
        uncom = elements[e]['used_uncommon']
        entity = 0.0
        for c in com:
            entity += com[c] * find_common_score(common_score, c)
        for u in uncom:
            entity += uncom[u] * find_common_score(common_score, u)
        if entity != 0:
            for c in com:
                score = freq * com[c] * find_common_score(common_score, c) / entity / (adj + 1)
                score *= find_common_score(common_score, c) # Disadvantage for common adverbs
                scoreboard.append((c, e, score))
            for u in uncom:
                score = freq * uncom[u] * find_common_score(common_score, u) / entity / (adj + 1)
                scoreboard.append((u, e, score))
        scoreboard.sort(key = lambda element: element[2])


def restrictive_score(scoreboard, accuracy_scoreboard, elements, dataset):
    '''
    1. Give penalty to general words for E
        a. General words would be used more frquently with many sort of common intensifiers
        b. However, difference in frequency and kinds can be cornered by
            some words that do not use either common or uncommon.
        c. Also, too sparse word can get small score. complement it by penalty.
    2. Give penalty to words that used with only commons
        a. It filters the phrase of common adverbs and sparse word.
    3. Give advantage to the phrase used frequently
        a. This means this phrase highly can be qualified to existing 'phrase'.
    '''
    cnt = 0
    for e in elements:
        cnt += 1
        freq = elements[e]['freq']
        common_freq = sum(elements[e]['used_common'].values())
        uncommon_freq = sum(elements[e]['used_uncommon'].values())
        common_kind = len(elements[e]['used_common'])
        uncommon_kind = len(elements[e]['used_uncommon'])
        # Words that can be scalable by commons, but also used restrictly by uncommons
        diff_score = (abs(common_freq - uncommon_freq) + 1) * (abs(common_kind - uncommon_kind) + 1)
        # Process it
        # If it is frequent but got low score, enhance it
        # If there is no usage with uncommons, deduct it
        # If it is too less frequent, deduct it
        if diff_score <= 10 and freq >= 100:
            diff_score = diff_score / float(freq)
        elif freq == 0 or uncommon_freq == 0:
            diff_score += 10.0
        elif freq <= 2:
            diff_score += 5.0
        for k in accuracy_scoreboard:
            specific_freq = find_freq(k[0], k[1], dataset)
            if k[1] == e:
                scoreboard.append((k[0], k[1], (k[2]+diff_score)/specific_freq))
    scoreboard.sort(key = lambda element: element[2])


def find_freq(D, E, dataset):
    return dataset[D]['used'][E]


def main():
    total_start_time = time.time()
    debug = False
    if debug:
        return

    visualize = False

    # Step 1. Extract raw info
    '''
    Step 1-1) Extract intensifier
    Method : Look up the all synsets in wordnet and picks synset that
    word 'intensifier' or 'degree' is well included in definition.
    And then append all lemmas of that synset.
    '''
    start_time = time.time()
    print("Extracting intensifiers..")
    intens = word_def_contains('intensifiers')
    degrees = filter_defs(word_def_contains('degree'))
    adverbs = list(set(intens + degrees))

    print("Intensifiers extracted, RUNTIME : %.3f" % (time.time() - start_time))
    print("Number of intensifiers : {}".format(len(adverbs)))

    # For visualization
    if visualize:
        print(adverbs)
    print()

    '''
    Step 1-2) See through bigrams and update adverbs information,
    1. Frequency they are used as ADJ/ADV in corpus
    2. Which lexical items are used with them in correct semantics(ADJ + NOUN, ADV + NOUN/ADV/VERB)
    3. Frequencies they are used with each lexical items.
    '''
    start_time = time.time()
    data = {}
    print("Extracting Data..")
    data_init(data, adverbs)
    update_data(data, adverbs)
    print("Data update completed, RUNTIME : %.3f" % (time.time() - start_time))
    print()

    '''
    Step 1-3) Discard unused intensifiers
    Investigate the data and delete unused intensifiers
    '''
    start_time = time.time()
    print("Discarding unused intensifiers...")
    bef = len(data)
    dataset = {}
    for i in data:
        if data[i]['freq'] != 0:
            dataset[i] = data[i]
    print("Discarding Done, RUNTIME : %.3f" % (time.time()-start_time))
    print("num of elements, from {} to {}".format(bef, len(dataset)))
    if visualize:
        for i in dataset:
            print(i, '->', dataset[i])
        print()
    print()

    # Step 2. Process bigrams
    '''
    Step 2-1) Extract common intensifiers
    Based on three criteria
    1. How many other ADJ/ADV definitions rather than sense of intensifiers?
    2. How many times it is used as ADJ or ADV?
    3. Rank of definition as intensifier
    '''
    start_time = time.time()
    common_score = []
    threshold = 1.0
    for adverb in dataset:
        score = get_commonity_score(adverb, dataset[adverb])
        common_score.append((adverb, score))
    common_score.sort(key = lambda element: element[1], reverse = True)
    uncommons = [adverb for adverb in common_score if adverb[1] <= threshold]
    commons = [adverb for adverb in common_score if adverb[1] > threshold]
    print("Scoring for commonity completed, RUNTIME : %.3f" % (time.time()-start_time))
    print("{} Common intensifiers".format(len(commons)))

    if visualize:
        print("Commons : {}".format(commons))
        print("Uncommons : {}".format(uncommons))
    print()

    '''
    Step 2-2) Build data for lexical items E
    For item E, following data would be in
    1. How many it is used in corpus?
    2. How adjective it is? (How many adjective definitions?)
    3. How many times it is used with common-intensifier?
    4. How many sort of common-intensifiers are used with it?
    5. How many times it is used with uncommon-intensifier?
    '''
    # Build dictionary for lexical items E
    start_time = time.time()
    elements = {}
    init_elements(elements, dataset)
    update_elements(elements, dataset, commons)
    print("Element update completed, RUNTIME : %.3f" % (time.time()-start_time))
    print("Size of Elements : {} entries".format(len(elements)))

    # Visualization
    if visualize:
        for i in elements:
            print(i, " -> ", elements[i])
    print()

    '''
    Step 2-3) Restriction score for each pair
    For each pairs of D and E, calculate restriction score
    LOW SCORE, HIGH RESTRICTION
    1st filter) Accuracy
    We would determine whether E is used with D in the really intensity modifying sense.
    + give penalty to common adverbs
    2nd filter) Restrictiveness
    Determine how restrictively D and E used.
    '''
    start_time = time.time()
    print("Restriction scoring ...")
    accuracy_scoreboard = []
    accuracy_score(accuracy_scoreboard, elements, common_score)
    scoreboard = []
    restrictive_score(scoreboard, accuracy_scoreboard, elements, dataset)
    print("Restriction scoring complete, RUNTIME : %.3f" % (time.time()-start_time))
    print()

    if visualize:
        cnt = 0
        for i in scoreboard:
            cnt += 1
            print(cnt , i)

    # Open the csv file, and make writer
    print("Generating Output...")
    f = open('CS372_HW2_output_20170490.csv', 'w', encoding='utf-8', newline='')
    csvwriter = csv.writer(f)
    for i in range(100):
        row = scoreboard[i][:2]
        csvwriter.writerow(row)
    print("Output is generated")

    # Write and Close the csv file
    f.close()

    print("Program terminated, RUNTIME : %.3f" % (time.time()-total_start_time))


main()
