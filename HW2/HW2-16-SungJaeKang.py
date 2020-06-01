#!/usr/bin/env python
# coding: utf-8

# In[53]:


import time
import re
from nltk.corpus import wordnet as wn
import csv

verbose = True


# In[54]:


def find_adverbs(word, compare):
    """ 
    Find intensity modifying adverbs from the definition 
    
    Given word and definition of the word, find possible intensity modifying adverb
    
    :param word: Word from which the function finds intensity-modifying adverb
    :type word: nltk.corpus.reader.wordnet.Synset
    :param compare:  Comparing word that may be in the definition of the 'word' parameter
    :type compare: string
    :return: Intensity-modifying adverbs from given word and definition
    :rtype: string, or None if any of adverbs found
    """
    
    # Compare should be different from the word definition
    if (word.name().split('.')[0] == compare):
        return None
    
    definition = word.definition().split()
    
    # Compare should be in the word definition
    if (compare not in definition):
        return None

    # Definition should be at least to informative to analyze further 
    # step1: skip when the definition contain negate words
    # since meaning of the word can be opposite to the compare
    # step2: skip when the definition contain redunddancy words
    # since there are no-informative words that can be found heuristically
    negate_words = {'no', 'not', 'without', 'lacking'}
    redundancy_words = {'more', 'in', 'up', 'on', 'away', 'by', 'be'}
    
    if (negate_words.intersection(definition)):
        return None
    if (redundancy_words.intersection(definition)):
        return None   
    
    idx = definition.index(compare)
    prev = definition[idx-1]
    cur = definition[idx]
    nxt = definition[(idx+1)%len(definition)]
    
    # Word should not be a noun, but there may happen due to the same spelling
    # Word(A) should not be used as adjective of the other noun(B) in the definition
    # since B isn't likely to have similar meaming with A.
    # Following steps remove these cases
    if (idx != 0 and (prev == "a" or prev == "the")):
        return None
    if (cur in nouns):
        return None
    if (cur.endswith("s") and cur[:-1] in nouns):
        return None
    if (cur.endswith("es") and cur[:-2] in nouns):
        return None
    
    if (nxt in nouns):
        return None
    if (nxt.endswith("s") and nxt[:-1] in nouns):
        return None
    if (nxt.endswith("es") and nxt[:-2] in nouns):
        return None

    # Return adverb that is found by the algorithm
    if (idx != 0 and prev in adverbs):
        return prev
    if (idx != len(definition)-1 and nxt in adverbs):
        return nxt
                
    return None

def find_word_triples(dictionary):
    """ 
    Find (word1, word2, intensity-modifying adverbs) Tuples from given dictionary
    
    From given dictionary, words will be obtained in following form;
    word1 = word2 + intensity-modifying adverb,
    that is, word1 will be intensity-modified word of word2

    :param dictionary: Dictionary from which the function finds intensity-modifying adverbs
    :type dictionary: Generator of nltk.corpus.reader.wordnet.Synset
    :return: Found triples of {word1, word2, intensity-modifying adverbs} Tuples
    :rtype: list of {word1: nltk.corpus.reader.wordnet.Synset, word2: string, adverb: string} Tuples
    
    """
    
    result = []
    
    dict_list = [w.name().split('.')[0] for w in dictionary]
    
    # Go through all possible word, compare pairs and find intensity modifying adverb
    for w in dictionary:
        for c in dict_list:
            adverb = find_adverbs(w, c)

            if (adverb != None):
                result.append((w, c, adverb))
    
    # Remove duplicates
    result = list(set(result))
    
    return result

def count_frequency(triples):
    """
    Count the frequency distribution of adverbs in the given triples
    
    :param triples: Triples to calculate the rank of given 'word2, adverb' pair
    :type triples: list of {word1: nltk.corpus.reader.wordnet.Synset, word2: string, adverb: string} tuples
    :return: Freqency distribution of adverbs
    :rtype: dict
    """
    
    result = {}
    
    for word1, word2, adverb in triples: 
        if (adverb in result): 
            result[adverb] += 1
        else: 
            result[adverb] = 1
        
    return result
        
def rank_triples(triples, alpha):
    """
    Rank the adverbs based on uniqueness and restrictiveness in decreasing order
    
    From given list of {word1: string, word2: string, adverb: string} tuples,
    analyze the uniqueness and restrictiveness of the adverb, and sort and return it 
    as in decreasing order of them
    
    I especially deviced new score function (frequency) + (number of words in definition)^2
    Uniqueness and restrictiveness can be obtained by choosing adverbs that rarely appears in
    the frequency distribution by nature, and confidence can be obtained where difinition has
    few words since there are highly likely to be few distracting words.
    
    :param triples: Triples to calculate the rank of given 'word2, adverb' pair
    :type triples: list of {word1: nltk.corpus.reader.wordnet.Synset, word2: string, adverb: string} tuples
    :param alpha: Hyper parameter of the score formula I made
    :type alpha: float
    :return: Found list of [word2, adverb, rank] sorted by the rank in decreasing order
    :rtype: list of [word2: string, adverb: string, rank: float]
    """
    
    # Make frequency distribution of adverbs
    freq_dict = count_frequency(triples)
    
    result = []
    
    for word1, word2, adverb in triples:
        l = len(word1.definition().split())
        f = freq_dict[adverb]
        
        score = l + alpha * f**2
        
        result.append([word1, word2, adverb, score])
        
    # sort by alphabetical order of adverb, as well as increasing order of score (decreasing order of rank)
    result = sorted(result, key=lambda result: result[2])
    result = sorted(result, key=lambda result: result[3])
        
    return result


# In[55]:


nouns = [w.name().split('.')[0] for w in wn.all_synsets('n')]
adverbs = [w.name().split('.')[0] for w in wn.all_synsets('r')]

if verbose: print("Make the adjective triples...")
start_time = time.time()
adjective_triples = find_word_triples(set(wn.all_synsets('a'))|set(wn.all_synsets('s')))
if verbose: print("{:.4f} seconds".format(time.time()-start_time))

if verbose: print("Make the verbs triples...")
start_time = time.time()
verb_triples = find_word_triples(set(wn.all_synsets('v')))
if verbose: print("{:.4f} seconds".format(time.time()-start_time))


# In[56]:


if verbose: print("Rank the combined triples...")
start_time = time.time()
combined_triples = adjective_triples + verb_triples
ranked_triples = rank_triples(combined_triples, 0.1)
if verbose: print("{:.4f} seconds".format(time.time()-start_time))


# In[ ]:


data = ranked_triples[:100]

if (verbose):
    for i in data:
        print(i)

csv_file = open("./CS372_HW2_output_20160009.csv", "w", newline="")

csv_writer = csv.writer(csv_file)

for row in data:
    csv_writer.writerow([row[2], row[1]])
    
csv_file.close()

