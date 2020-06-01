#!/usr/bin/env python
# coding: utf-8

import nltk
from urllib import request
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
import re

from wiktionaryparser import WiktionaryParser
from nltk.stem.wordnet import WordNetLemmatizer

import pprint
from collections import defaultdict

import csv

import time

verbose = True

urls =  ["http://jonv.flystrip.com/heteronym/smithsonian.htm",
         "http://jonv.flystrip.com/heteronym/heteronym.htm",
         "http://www.heteronym.com/what_is_a_heteronym__m.htm",
         "http://www.heteronym.com/heteronymphs_m.htm",
         "http://www.heteronym.com/heteronymical_history_m.htm",
         "http://www.heteronym.com/the_case_of_the_lady_who_wasn_t_m.htm",
         "http://www.heteronym.com/are_you_content__m.htm"]

def find_heteronyms(urls):
    """Find heteronym candidates from source corpora.

    Instructions:
        1. Getting raw text from source corpus.
        2. Divide text to sentences.
        3. For each sentence, find word that appears more than twice in the sentence.
        4. check whether the word has more than two pronunciations, then set it heteronym CANDIDATES if so.

    Args:
        1. urls (list): List of strings which contain source corpora urls.
        
    Returns:
        heteronym candidates (list): List of heteronym candidates.
                                    (not determined in this step in order to  contain homogragh as well).
    """
    cmudict = nltk.corpus.cmudict.dict()

    heter = []

    for url in urls:
        html = request.urlopen(url).read()
        raw = BeautifulSoup(html, "html.parser").get_text()
        sents = sent_tokenize(raw)

        for sent in sents:
            text = nltk.word_tokenize(sent)
            tagged_words = nltk.pos_tag(text)

            words = [word[0].lower() for word in tagged_words]
            tags = [word[1] for word in tagged_words]

            duplicate = list(set([word for word in words if words.count(word) > 1]))

            for word, tag in zip(words, tags):
                if (word in cmudict and len(cmudict[word]) > 1 and word in duplicate
                    and ((re.fullmatch(r'NN', tag)) or (re.fullmatch(r'VB', tag)) or (re.fullmatch(r'JJ', tag)))):
                    heter.append([word, tag])
                    
    heter = [w[0] for w in heter]
    heter = list(set(heter))
    heter.sort()
    
    return heter

def generate_model(heter):
    """Generate the model which maps (heteronym word) to (context, pronunciation).
    
    Context consist of syntatical and semantical context, from which the algorithm
    can determine which heteronym is often used in certain situation.
    
    Syntatical context : 
    POS of the heteronym, 
    POS of neighboring words of the heteronym, which is mostly appeard considering 
    multiple sentences which contain the heteronym.
    
    Semantical context :
    Set of words that appeared in the example sentences which contain the heteronym.
    
    Note. pronunciation is IAP form since I utilized Wiktionary for building the model.
    
    model
    |-- word1
        |-- POS
        |-- most frequent POS of neighboring words
        |-- Set of semantical words set representing the word
        `-- Pronunciation
    |-- word2
        |-- POS
        |-- most frequent POS of neighboring words
        |-- Set of semantical words set representing the word
        `-- Pronunciation
    |-- ...
    `-- wordk
        |-- POS
        |-- most frequent POS of neighboring words
        |-- Set of semantical words set representing the word
        `-- Pronunciation

    Instructions:
        1. From heteronym candidates, use Wiktionary to check whether the candidate has more than
           two different meanings (based on definition of heteronym).
        2. Obtain possible example sentences related to each heteronym candidata.
        3. From the sentences, extract SYNTATICAL and SEMANTICAL contexts and put into the model.
        4. Also pronunciation information added to the model for prediction step.

    Args:
        1. hetero (list): List of heteronym candidates.
        
    Returns:
        model (dict): model to predict pronunciation based on word spelling and relative context.
    """
    model = defaultdict(list)
    
    parser = WiktionaryParser()

    for het in heter:
        words = parser.fetch(het)

        for i, word in enumerate(words):
            pos = []
            for var in word['definitions']:
                pos.append(var['partOfSpeech'])
                break
                
            sents = []
            for var in word['definitions']:
                sents = sents + var['text']
                sents = sents + var['examples']

            pattern = r'IPA'
            pron = ""
            cnt = 0
            for var in word['pronunciations']['text']:
                match = re.findall(r'IPA.*', var)

                if (match):
                    pron = match[0].split(" ")[1].replace("/", "")
                    if (i == cnt):
                        break
                    else:
                        cnt += 1

            cxt1 = semantic_context(sents)
            cxt2 = syntactic_context(sents, het)

            info = [pos, cxt1, pron, cxt2]

            model[het].append(info)
            
    for key in model:
        model[key] = sorted(model[key], key=lambda e: e[2])
            
    return model

def semantic_context(sents):  
    """Generate semantical context from sentences.
    
    Instructions:
        1. Parsing the sentences into words.
        2. Remove unnecessary words as well as making them into lower case.
        3. Lemmatize twice for 1) make words into original form 2) same for verb case.
        4. Remove duplicates.

    Args:
        1. sents (list): List of sentences.
        
    Returns:
        semantic context (set): Set of SEMANTICAL context.
    """
    context = [nltk.word_tokenize(sent) for sent in sents]
    context = [nltk.pos_tag(sent) for sent in context]
    context = [w for sent in context for w in sent]
    context = [(w[0].lower(), w[1]) for w in context if w[0].isalpha()] 
    context = [(WordNetLemmatizer().lemmatize(w[0]), w[1]) for w in context]
    context = context + [(WordNetLemmatizer().lemmatize(w[0],'v'), w[1]) for w in context]
    context = [w[0] for w in context]
    context = set(context)
    
    return context

def syntactic_context(sents, het):
    """Generate syntatical context from sentences and the word.
    
    Instructions:
        1. Parsing the sentences into words.
        2. Does POS tagging and find POS of neighboring words of the heteronym.

    Args:
        1. sents (list): List of sentences.
        2. het (string): Heteronym candidate word.
        
    Returns:
        syntactic context (set): Set of SYNTACTICAL context.
    """
    result = []
    
    context = [nltk.word_tokenize(sent) for sent in sents if het in sent.split()]
    context = [nltk.pos_tag(sent) for sent in context]
    for token in context:
        sent = [s[0] for s in token]
        tag = [s[1] for s in token]
        
        idx = sent.index(het)
        
        idx1 = (idx-2)%len(sent)
        idx2 = (idx-1)%len(sent)
        idx3 = (idx+1)%len(sent)
        idx4 = (idx+2)%len(sent)
        
        result.append([tag[idx1], tag[idx2], tag[idx3], tag[idx4]])
        
    result = [max(set(l), key = l.count) for l in list(zip(*result))]
    
    return result

def analyze_corpus(model, urls, w1, w2, w3):
    """Analyze the corpora to give pronunciation to each heteronym.

    Note that I regard the word which has more than twice as HETERONYM following the definition
    of the HETERONYM. The reason why I didn't filter it on the early stage is to leave the 
    HOMOGRAPH, as it is used when giving priority to sort following the ranking instruction
    of the official document.

    Instructions:
        1. From the corpora, parse them into sentences.
        2. For each sentence, ckech whether heteronyms exist in the sentence.
        3. if the heteronym exist, score the heteronym using context of the sentence.

    Args:
        1. model (dict): model to predict pronunciation based on word spelling and relative context.
        2. urls (list): List of strings which contain source corpora urls.
        3. w1, w2, w3 (float): weight given to each similarity with heteronym.
                               w1: POS of the word weight.
                               w2: Sementical context weight.
                               w3: POS of neighboring words weight.
        
    Returns:
        result (list): extracted heteronym with (pos, pronunciation, source sentence, cite)
                       as well as three priorities listed in the problem statment document to sort them.
    """
    result = []

    for url in urls:
        html = request.urlopen(url).read()
        raw = BeautifulSoup(html, "html.parser").get_text()
        sents = sent_tokenize(raw)

        for sent in sents:
            # Extract possible outliers
            if (len(sent) > 150):
                continue
            
            sent_heteronym = []
            sent_pron = []
            sent_pos = []

            text = nltk.word_tokenize(sent)

            text = [w.lower() for w in text]
            tagged_words = nltk.pos_tag(text)

            # candidate extranction
            candidates = []
            contexts = []
            semantic_contexts = []
            idxs = []

            for idx, tagged_word in enumerate(tagged_words):  
                word = tagged_word[0]
                pos = tagged_word[1]

                if (len(model[word]) >= 2):
                    candidates.append(tagged_word)

                    # context generation.
                    idx1 = (idx-2)%len(text)
                    idx2 = (idx-1)%len(text)
                    idx3 = (idx+1)%len(text)
                    idx4 = (idx+2)%len(text)

                    # Lemmatizer
                    stemmered_text = [WordNetLemmatizer().lemmatize(w) for w in text]
                    stemmered_text = text

                    context = [stemmered_text[idx1], stemmered_text[idx2], stemmered_text[idx3], stemmered_text[idx4]]
                    contexts.append(context)
                    
                    semantic_context = [tagged_words[idx1][1], tagged_words[idx2][1], tagged_words[idx3][1], tagged_words[idx4][1]]
                    semantic_contexts.append(semantic_context)
                    
                    idxs.append([idx1, idx2, idx3, idx4])

            # giving candidates proper pronunciations 
            noun_pattern = r'NN'
            adj_pattern = r'JJ'
            verb_pattern = r'VB'
            flag = True
            
            for (idx, cand, context, semantic_context) in zip(idxs, candidates, contexts, semantic_contexts):
                word = cand[0]
                pos = cand[1]

                # if POS is noun
                if (re.fullmatch(noun_pattern, pos)):
                    pos = "noun"

                # if POS is adjective
                elif (re.fullmatch(adj_pattern, pos)):
                    pos = "adjective"

                # if POS is verb
                elif (re.fullmatch(verb_pattern, pos)):
                    pos = "verb"
                    
                else:
                    flag = False
                    break

                max_score = -1
                pron = ""
                for cmp in model[word]:
                    score = 0

                    if (pos in cmp[0]):
                        score += w1

                    for cxt in context:
                        if (cxt in cmp[1]):
                            score += w2
                    
                    for i, (cxt, ix) in enumerate(zip(semantic_context, idx)):
                        if (cxt == cmp[3][i]):
                            score += w3

                    if (score > max_score):
                        max_score = score
                        pron = cmp[2]

                sent_heteronym.append(word)
                sent_pron.append(pron)
                sent_pos.append(pos)

            # give priority to sort them in given order
            prior_first = len(candidates)
            prior_second = len(set(candidates))
            prior_third = 0
            unique_word_pos = []
            for (w, p) in zip(sent_heteronym, sent_pron):
                if ([w, p] not in unique_word_pos):
                    prior_third += 1
                    unique_word_pos.append([w, p])

            # Create final result
            if (sent_heteronym and flag):
                result.append([prior_first, prior_second, prior_third, sent, sent_heteronym, sent_pron, sent_pos, url])

    return result

if verbose: print("Make the heteronym candidates...")
start_time = time.time()
heteronyms = find_heteronyms(urls)
if verbose: print("{:.4f} seconds".format(time.time()-start_time))

if verbose: print("Generating model...")
start_time = time.time()
model = generate_model(heteronyms)
if verbose: print("{:.4f} seconds".format(time.time()-start_time))

if verbose: print("Analyze corpora...")
start_time = time.time()
result = analyze_corpus(model, urls, 4.2, 1.4, 2.2)
if verbose: print("{:.4f} seconds".format(time.time()-start_time))

result = sorted(result, key = lambda x: x[2])
result = sorted(result, key = lambda x: x[1])
result = sorted(result, key = lambda x: x[0], reverse = True)
result = [r[3:] for r in result]

data = []
for e in result[:30]:
    temp = []
    
    sent = e[0]
    words = e[1]
    prons = e[2]
    poss = e[3]
    cite = e[4]
    
    temp.append(sent)
    for (a, b, c) in zip(words, prons, poss):
        temp.append(a)
        temp.append(b)
        temp.append(c)
    temp.append(cite)
    
    data.append(temp)

csv_file = open("./CS372_HW3_output_20160009.csv", "w", encoding="utf-8-sig", newline="")

csv_writer = csv.writer(csv_file)

for row in data:
    csv_writer.writerow(row)
    
csv_file.close()

