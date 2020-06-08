import nltk
import time  # For estimate time complexity
import math  # For score system
import csv  # For output
import os.path  # For storing data, which take much time at birth CONVENIENCE
import pickle  # For storing data, which take much time at birth CONVENIENCE
import random  # For diverse phrase
import re
import sys
import urllib.error
from nltk.corpus import brown
from nltk.corpus import reuters
from nltk.corpus import inaugural
from nltk.corpus import wordnet as wn
from nltk.corpus import cmudict
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from urllib.request import urlopen
from bs4 import BeautifulSoup

"""
It includes some development environment, code pieces for debugging or convenience
This is not released form.
I would left this to let the instructors know my intention well...!
"""
class PosTagError(Exception):
    """Exception raised when incorrect pos tag is passed"""
    def __init__(self, message):
        super(PosTagError, self).__init__(message)
        self.message = message

    # __str__ is to print() the value
    def __str__(self):
        return(repr(self.message))


class Dict_entry():
    '''
    Entry of dictionary.
    For each entry, it has pronunciations.
    For each pronunciation, it has representative pos and def usage.
    It gets information from Dictionary.com
    '''
    def __init__(self, word):
        self.entry = word
        self.pron = []
        self.pos_set = []
        self.def_set = []
        self.update()

    def __str__(self):
        tab = "    "
        if len(self.pron) != len(self.pos_set) != len(self.def_set):
            return "Invalid word data"
        result = "{}:\n".format(self.entry)
        for i in range(len(self.pron)):
            result += tab + "({})\n".format(self.pron[i])
            for j in range(len(self.pos_set[i])):
                result += tab*2 + "{} : {}\n".format(self.pos_set[i][j], self.def_set[i][j])
        return result

    def valid_entry_check(self):
        """
        Check if input is null or contains only spaces or numbers or special characters
        """
        temp = re.sub(r'[^A-Za-z ]', ' ', self.entry)
        temp = re.sub(r"\s+", " ", temp)
        temp = temp.strip()
        if temp != "":
            return True
        return False

    def update(self):
        """
        Fetches the prons/pos_set/definitions from the dictionary.com \
        using Beautiful Soup
        """
        try:
            if self.valid_entry_check():
                response = urlopen(
                    'http://www.dictionary.com/browse/{}'.format(self.entry))
                html = response.read().decode('utf-8')
                soup = BeautifulSoup(html, 'lxml')
                prons = []
                pos_set = []
                def_set = []

                h2s = soup.find_all('h2', id='collins-section')
                for h2 in h2s:
                    get_pron_from_chunk = False
                    section = h2.next_sibling
                    if section.select('span.e1rg2mtf8')[0].text != self.entry:
                        continue
                    pron = section.select('div.e1rg2mtf6')[0].select('span.pron')
                    if len(pron) > 1: # It has multiple prons but same meaning
                        continue
                    elif len(pron) == 0: # It's prons is not shown next to the word. It is shown in each chunk
                        get_pron_from_chunk = True
                    # pos = section.select('section.e1hk9ate0 > h3.e1hk9ate1 > span.e1hk9ate2 > span.pos')
                    chunk = section.select('section.e1hk9ate0')
                    pos = []
                    defs = []
                    for ch in chunk:
                        if get_pron_from_chunk:
                            temp = ch.find('span', 'pron')
                            if temp is not None:
                                pron.append(temp) # get the first pron of that meaning
                            else:
                                continue
                        temp = ch.find('span', 'pos')
                        if temp is not None:
                            pos.append(temp)
                        temp = ch.find('div', 'e1q3nk1v3')
                        d = ", ".join([t.text for t in temp.find_all('span', 'e1q3nk1v4')])
                        defs.append(d)
                    # pron[:] = [p for p in pron if p is not None]
                    if len(pron) > 1: # This pron is from the chunks
                        prons += [p.text for p in pron]
                        pos_set += [[p.text] for p in pos]
                        def_set += [[d] for d in defs]
                    elif len(pron) == 1:
                        prons.append(pron[0].text)
                        pos_set.append([p.text for p in pos])
                        def_set.append(defs)

                self.pron = prons
                self.pos_set = pos_set
                self.def_set = def_set
            else:
                print("Provide a not-null input word")
                return
        except urllib.error.HTTPError as err:
            if err.code == 404:
                # print("Word is not valid")
                return
        except IndexError:
            print("Give a non-empty argument")
            return
        except urllib.error.URLError:
            print("No Internet Connection")
            return


# Debug functions
def print_def(word, token = ' '):
    tab = '    '
    print("Definitions of {}".format(word))
    for synset in wn.synsets(word):
        if token in synset.definition():
            print(tab + "{} : {}".format(synset, synset.definition()))


def print_entry_debug(entry):
    print(entry.entry)
    print(entry.pron)
    print(entry.pos_set)
    print(entry.def_set)


# Body functions
def get_nice_words(corpus):
    '''
    Get words which is lower, alpha, and not stopwords from corpus
    return: list
    '''
    words = set()
    stopWords = stopwords.words('english')
    for fileid in corpus.fileids():
        words.update(corpus.words(fileid))
    return [word for word in words if word not in stopWords
            and word.isalpha() and word.islower()]


def synset_filter(words):
    '''
    Discard words that have only one pos
    return: List[String]
    '''
    return [word for word in words if is_synset_many_pos(word)]


def is_synset_many_pos(word):
    '''
    Whether the word can have multiple pos
    :param word: String
    :return: boolean
    '''
    pos = set()
    for synset in wn.synsets(word):
        p = synset.pos()
        if p == 's':
            pos.add('a')
        else:
            pos.add(p)
    if len(pos) > 1:
        return True
    else:
        return False


def cmudict_filter(words):
    '''
    Discard words that have single pronunciation
    :return: List[String]
    '''
    result = []
    prondict = nltk.corpus.cmudict.dict()
    for word in words:
        try:
            if len(prondict[word]) >= 2:
                result.append(word)
        except KeyError:
            continue
    return result


def crawl_heteronyms(candidates):
    '''
    1) Search the candidates in Dictionary.com
    2) Parse the response to get their pronunciations, pos, and definition.
    3) Organize those information in Dict_entry class.
    return: dictionary of heteronyms
    '''
    myHeteroDict = {}
    length = len(candidates)
    count = 0
    start_time = time.time()
    for word in candidates:
        count += 1
        entry = Dict_entry(word)
        if len(set(entry.pron)) > 1:
            myHeteroDict[entry.entry] = entry
        if count % 100 == 0:
            print("{} words done per {} words : RUNTIME {}s".format(
                count, length, round(time.time() - start_time, 3)
            ))
            start_time = time.time()
    return myHeteroDict


def harvest_sentence(corpus_list):
    '''
    Harvest whole sentences from every corpus
    :param corpus_list: list reference to each corpus
    :return: List of sentences (Huge!)
    '''
    corpus_name_list = ['nltk.brown corpus', 'nltk.reuters corpus', 'nltk.inaugural corpus']
    result = []
    i = 0
    for corpus in corpus_list:
        for fileid in corpus.fileids():
            for sent in corpus.sents(fileid):
                entry = sent + [corpus_name_list[i]]
                result.append(entry)
        i += 1
    return result


def investigate_usage(sentences, heteronyms):
    '''
    Investigate the usage of heteronyms in each sentences
    and get data attributes.
    List of {'hets': (heteronyms, pos usage), sent: sentence, cited: cited}
    :param sentences: Every sentences in corpus
    :param heteronyms: heteronyms from myHeteroDict
    :return: List of dictionaries
    '''
    data = []
    for sent_data in sentences:
        sent = sent_data[:-1]
        cited = sent_data[-1]
        hets = []
        entry = {}
        target = False
        for word in sent:
            if word in heteronyms:
                target = True
                break
        if target:
            '''
            Here is trade off between processing time and performance.
            We do lemmatization for only the sentences which has a heteronym as original shape,
            in order to not miss the chance of finding homograph in that sentence.
            We don't do pos_tagging and lemmatization for every sentence. It takes too much time.
            '''
            tagged_sent = nltk.pos_tag(sent)
            for word, tag in tagged_sent:
                word_lem, tag_lem = lemmatize((word, tag))
                if word_lem in heteronyms:
                    hets.append({'name': word_lem, 'pos': tag, 'pron': ""})
            entry['hets'] = hets
            entry['sent'] = ' '.join(sent)
            entry['cited'] = cited
            data.append(entry)
    return data


def lemmatize(b):
    '''
    Lemmatize lexical items
    For NOUN : Change plural to singular,
    For VERB : Change to original,
    For ADJ : Change to original,
    For ADV : Change to original,
    '''
    wnl = WordNetLemmatizer()
    ADJ = ['JJR', 'JJS', 'JJT']
    NOUN = ['NNS'] # Ignore 'NP' 'NPS' 'NR'
    VERB = ['VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    ADV = ['RBR', 'RBT', 'RN', 'RP']
    if b[1] in NOUN:
        lemmatized = wnl.lemmatize(b[0], 'n')
        return lemmatized, 'NN'
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


def translate_pos(pos):
    '''
    Translate tags of pos_tag to universal pos in dictionary.com
    :param pos: pos_tag tag
    :return: universal pos
    '''
    ADJ = ['JJ', 'JJR', 'JJS', 'JJT']
    NOUN = ['NN', 'NNS'] # Ignore 'NP' 'NPS' 'NR'
    VERB = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    ADV = ['RBR', 'RBT', 'RN', 'RP']
    if pos in ADJ:
        return 'adjective'
    elif pos in NOUN:
        return 'noun'
    elif pos in VERB:
        return 'verb'
    elif pos in ADV:
        return 'adverb'
    else:
        return pos


def check_pos_exist(pos, entry):
    '''
    Check whether the entry.pos_set contains the following pos
    else, substitute it.
    :param pos: pos
    :param entry: Dict_entry
    :return: pos (substituted)
    '''
    pos_set = set()
    for i in entry.pos_set:
        for j in i:
            pos_set.add(j)
    if pos not in pos_set:
        if pos == 'adjective':
            return 'noun'
        elif pos == 'verb':
            return 'adjective'
        elif pos == 'noun':
            return 'adjective'
    return pos


def calculate_similarity(pos, definition, sent):
    '''
    Calculate the similarity between definition and sentence,
    to decide the definition is really for the sentence.
    :param pos: pos of candidate
    :param definition: definition of candidate
    :param sent: target sentence
    :return: score 0~1
    '''
    ADJ = ['JJ', 'JJR', 'JJS', 'JJT']
    NOUN = ['NN', 'NNS'] # Ignore 'NP' 'NPS' 'NR'
    VERB = ['VB', 'VBD', 'VBG']
    ADV = ['RBR', 'RBT', 'RN', 'RP']
    tagged_sent = nltk.pos_tag(nltk.word_tokenize(sent))
    tagged_def = nltk.pos_tag(nltk.word_tokenize(definition))
    keyword_def = []
    keyword_sent = []
    if pos == 'adjective':
        keyword_def = [word[0] for word in tagged_def if word[1] in ADJ]
        keyword_sent = [word[0] for word in tagged_sent if word[1] in ADJ]
    elif pos == 'noun':
        keyword_def = [word[0] for word in tagged_def if word[1] in ['NN', 'NNS']]
        keyword_sent = [word[0] for word in tagged_sent if word[1] in NOUN]
    elif pos == 'verb':
        keyword_def = [word[0] for word in tagged_def if word[1] is 'VB']
        keyword_sent = [word[0] for word in tagged_sent if word[1] in VERB]
    elif pos == 'adverb':
        keyword_def = [word[0] for word in tagged_def if word[1] in ADV]
        keyword_sent = [word[0] for word in tagged_sent if word[1] in ADV]
    else:
        return 0
    sum = 0.0
    cnt = 1  # To avoid zero-division

    for dkey in keyword_def:
        dsets = wn.synsets(dkey)
        if len(dsets) == 0:
            continue
        dset = dsets[0]  # Assume that dictionary definition uses most representative word
        for skey in keyword_sent:
            ssets = wn.synsets(skey)
            if len(ssets) == 0:
                continue
            sset = ssets[0]
            if sset is None:
                continue
            cnt += 1
            try:
                sum += dset.path_similarity(sset)
            except TypeError:
                sum += 0.0
    return sum/float(cnt)


def make_set(data):
    '''
    When the data is unhashable, compare it directly and generate set
    :param data: list of unhashables
    :return: list of unhashables, duplicates discarded
    '''
    result = []
    for d in data:
        temp = True
        for r in result:
            if d == r:
                temp = False
                break
        if temp:
            result.append(d)
    return result


def calculate_score(entry):
    '''
    Calculate score for each entry
    Entries are including the critics
    :param entry: entry in data
    :return: int score
    '''
    # Assume the ranking criteria is lower that 10
    _1 = len(entry['homographs'])
    try:
        _2 = max([e['num'] for e in entry['homographs']])
    except ValueError:
        _2 = 0
    _3 = entry['heteronyms']
    _4 = entry['same_pos']
    _5 = len(entry['hets'])
    _6 = len(entry['sent'])  # Assume there is no sentence which length is bigger than one thousands.
    return _1 * 100000 + _2 * 10000 + _3 * 1000 + _4 * 100 + _5 * 10 - _6 * 0.01


def main():
    total_start_time = time.time()
    debug = False  # Run tiny debug code and quit
    visualize = False  # visualize the result in console log
    backup = True  # Back up the intermediate date by pickle
    if debug:
        print_def('bass')
        return

    # Heteronym dataset
    myHeteroDict = {}
    data = []
    corpus = brown

    if backup and not os.path.isdir('tempData20170490JaejunLee'):
        os.mkdir("tempData20170490JaejunLee")
    # Step 1. Find heteronyms and build dictionary
    '''
    Step 1-1 : Filter out candidates for heteronyms from corpus
    Filter 1 : length filter - Long words would have less flexibility in meaning and pronuns.
    Filter 2 : cmudict filter - Homographs would have multiple pronuns entries in the cmudict.
    Filter 3 : synset filter - Homographs would contain multiple pos usage in their synsets. 
    '''
    print("Extract heteronym candidates from corpus...")
    start_time = time.time()
    words = get_nice_words(corpus)
    print("raw words : ", len(words))
    words[:] = [word for word in words if len(word) <= 10]
    print("length filtered : ", len(words))
    words = cmudict_filter(words)
    print("cmudict filtered : ", len(words))
    words = synset_filter(words)
    print("synset filtered : ", len(words))
    print("Extraction complete, {} entries, RUNTIME {}s\n"
          .format(len(words), round(time.time()-start_time, 3)))

    '''
    Step 1-2 : Crawl the heteronym dictionary from Dictionary.com
    If the pickle is ready, load it.
    Else follow the steps.
    '''
    print("Generate heteronym dictionary...")
    crawling_time = time.time()
    if os.path.isfile('tempData20170490JaejunLee/myHeteroDict.txt'):
        print("We load existing heteronyms...")
        f = open('tempData20170490JaejunLee/myHeteroDict.txt', 'rb')
        myHeteroDict = pickle.load(f)
        f.close()
    else:
        print("We generate heteronyms from online dictionary... PLEASE WAIT")
        myHeteroDict = crawl_heteronyms(words)
        if backup:
            f = open('tempData20170490JaejunLee/myHeteroDict.txt', 'wb')
            pickle.dump(myHeteroDict, f)
            f.close()
    print("Heteronym dictionary generation completed, {} entries, RUNTIME : {}s\n"
          .format(len(myHeteroDict), round(time.time()-crawling_time, 3)))

    myHeteronyms = myHeteroDict.keys()

    if visualize:
        for key in myHeteroDict.keys():
            try:
                print(myHeteroDict[key])
            except:
                print("###### Debugging Dictionary.com parser ######")
                print_entry_debug(myHeteroDict[key])
            print()

    # Step 2. Harvest sentences that using the heteronyms
    if os.path.isfile('tempData20170490JaejunLee/hw3data.txt'):
        print("We load existing sentence data...")
        f = open('tempData20170490JaejunLee/hw3data.txt', 'rb')
        data = pickle.load(f)
        f.close()
    else:
        '''
        Step 2-1 : Get the whole sentence in corpus
        Simple iteration through corpus
        '''
        print("Harvest sentences...")
        corpus_list = [brown, reuters, inaugural]
        start_time = time.time()
        sentences = harvest_sentence(corpus_list)
        print("Harvest complete, {} sentences from {} corpus, RUNTIME : {}s\n"
              .format(len(sentences), len(corpus_list), round(time.time()-start_time, 3)))

        '''
        Step 2-2 : Investigate usage of heteronyms for each sentence
        '''
        print("Investigate heteronym usage...")
        data = investigate_usage(sentences, myHeteronyms)
        print("Investigation complete, {} sentence data, RUNTIME : {}s\n"
              .format(len(data), round(time.time()-start_time, 3)))
        data.sort(key=lambda entry: len(entry['hets']))
        data.reverse()

        if backup:
            f = open('tempData20170490JaejunLee/hw3data.txt', 'wb')
            pickle.dump(data, f)
            f.close()

    if visualize:
        for d in data[:300]:
            print(d)
    # Step 3. Confirm the pronunciation and pos of each heteronyms
    '''
    Step 3-1: Find myHeteroDict to get pronunciation candidates for each heteronym
    Then, pick a right pronunciation among the candidates
    '''
    print("Commit the pronunciation for heteronyms...")
    start_time = time.time()
    for entry in data:
        for het in entry['hets']:
            name = het['name']
            pos = translate_pos(het['pos'])
            value = myHeteroDict[name]
            pos = check_pos_exist(pos, value)
            cand = []
            for i in range(len(value.pron)):
                for j in range(len(value.pos_set[i])):
                    if value.pos_set[i][j] == pos:
                        cand.append({'pron': value.pron[i], 'pos': value.pos_set[i][j], 'def': value.def_set[i][j], 'rank': j+1})
            '''
            Step 3-2 : Confirm the pronunciation and pos of heteronym
            If there are only one candidates, commit it
            else,
                get the rank of definition. rank of definition means the order of definition for that pronuns.
                Calculate how similiar the words in definition and the words in sentence.
                Similarity/rank will be the critic for concluding the commit of pronuns
            '''
            if len(cand) == 1:
                het['pron'] = cand[0]['pron']
                het['pos'] = cand[0]['pos']
            elif len(cand) > 1:
                for c in cand:
                    c['sim'] = calculate_similarity(c['pos'], c['def'], entry['sent'])
                cand.sort(key=lambda entry: c['sim']/c['rank'])

                het['pron'] = cand[0]['pron']
                het['pos'] = cand[0]['pos']
            else:
                entry['debug'] = pos

    print("Pronunciation commit complete, RUNTIME : {}s\n"
          .format(round(time.time() - start_time, 3)))

    if visualize:
        for d in data[:300]:
            print(d)


    # Step 4. Ranking
    '''
    Step 4-1 : Parse the data and Get ranking critics.
    What is homograph? : group of two or more heteronyms, different each other
    1) If the sentence contains multiple homographs, higher rank. -> # of homographs
    +) If one homograph has multiple heteronyms, higher rank.
    2) If the sentence contains multiple heteronyms (not homographs), higher rank
    Ex) wind(N) + wind(V) > wind(N) + tear(V) + use(V) > wind(N) + tear(V) > wind(N)
    3) If heteronyms in homograph has same pos but different meaning, higher rank.
    Ex) bass(N, voice) + bass(N, fish) > bass(N, fish) + bass(Adj, voice)
    Investigate those and insert to data entry
    '''
    print("Ranking the sentences...")
    start_time = time.time()
    for entry in data:
        het_set = make_set(entry['hets']) # Make heteronym set
        entry['heteronyms'] = len(het_set) # The number of kinds of heteronyms
        names = [het['name'] for het in het_set] # Get names of heteronyms
        name_set = set(names) # Get the set of names
        name_pos_list = [] # pos of each heteronym
        entry['homographs'] = []
        for name in name_set: # Search the heteronyms with the name
            homs = names.count(name) # Get how many same names are in heteronyms, which means homograph
            if homs >= 2: # If there are more than 2, consider it as homograph
                entry['homographs'].append({'name': name, 'num': homs})
            name_pos_list = [het['pos'] for het in het_set if het['name'] == name]
        entry['same_pos'] = len(name_pos_list) - len(set(name_pos_list))

    '''
    Step 4-2: Calculate rank of each sentence based on the critiques, and sort it
    '''
    for entry in data:
        entry['score'] = calculate_score(entry)
    data.sort(key=lambda entry: entry['score'])
    data.reverse()
    print("Ranking complete, RUNTIME : {}s\n"
          .format(round(time.time() - start_time, 3)))

    '''
    Step 4-3: Discard duplicate sentences.
    '''
    ranked = []
    count = 0
    for entry in data:
        if len(ranked) >= 1 and ranked[-1]['sent'] == entry['sent']:
            continue
        count += 1
        ranked.append(entry)
        if count == 30:
            break

    if visualize:
        for d in ranked:
            print(d)

    # Open the csv file, and make writer
    print("Generating Output...")
    f = open('CS372_HW3_output_20170490.csv', 'w', encoding='utf-8', newline='')
    csvwriter = csv.writer(f)
    rank = 1
    for entry in ranked:
        hets = []
        for het in entry['hets']:
            het_string = "{}({},{})".format(het['name'], het['pos'], het['pron'])
            hets.append(het_string)
        row = [entry['sent']] + hets + [entry['cited']]
        csvwriter.writerow(row)
        rank += 1
    print("Output is generated")

    # Write and Close the csv file
    f.close()

    print("Program terminated, RUNTIME : %.3f" % (time.time()-total_start_time))


main()
