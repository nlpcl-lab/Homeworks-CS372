#!/usr/bin/env python
# coding: utf-8

# In[1]:


from nltk.corpus import wordnet as wn
from collections import defaultdict
from nltk import bigrams
from nltk.util import ngrams
import csv
import random

# For reproductivity of the result
# And also by shuffling the result, I can show that the result in a various words
# not in the alpabetical order, and show the result is generally nice as well
random.seed(0)

words_by_pos = defaultdict(list)

for synset in wn.all_synsets():
    pos = synset.pos()
    words_by_pos[pos].append(synset)
    
# find all set of adjectives, verbs, nouns, adverbs from wordnet        
all_adjectives = list(words_by_pos['a']) + list(words_by_pos['s']) 
random.shuffle(all_adjectives)
all_adjectives_name = [w.name().split('.')[0] for w in all_adjectives]

all_verbs = list(words_by_pos['v'])
random.shuffle(all_verbs)
all_verbs_name = [w.name().split('.')[0] for w in all_verbs]

all_nouns = list(words_by_pos['n'])
all_nouns_name = [w.name().split('.')[0] for w in all_nouns]

all_adverbs = list(words_by_pos['r'])
all_adverbs_name = [w.name().split('.')[0] for w in all_adverbs]


# In[2]:


# find intensity modifying adverbs from detected adjective or verb pairs
# definition : list of string
# word :  string
def find_adverbs(definition, word):
    for (i, w) in zip(range(len(definition)), definition):
        if (word == w):
            if (i == 0 and len(definition) != 1):
                if (definition[1] in all_adverbs_name):
                    return definition[1]
            elif (i == len(definition)-1):
                if (definition[i-1] in all_adverbs_name):
                    return definition[i-1]
            else:
                if (definition[i+1] in all_adverbs_name):
                    return definition[i+1]
                if (definition[i-1] in all_adverbs_name):
                    return definition[i-1]
    return None


# In[3]:


adjectives_pair = []

for w in all_adjectives:
    w_name = w.name().split('.')[0]
    for c in all_adjectives_name:
        
        # Continue when w == c
        if (w_name == c):
            continue
            
        definition = w.definition().split()
        
        # Negate word detection
        # This kind of words change the meaning, so I took out those cases
        if ('no' in definition):
            continue
        elif ('not' in definition):
            continue
        elif ('without' in definition):
            continue
        elif ('lacking' in definition):
            continue
            
        # Unnecessary word detection
        # some words are both adjective and adverb, especillay frequent in most of definitions
        # so I took out those cases
        elif ('more' in definition):
            continue
        elif ('in' in definition):
            continue
        elif ('up' in definition):
            continue
        elif ('on' in definition):
            continue
        elif ('away' in definition):
            continue
        elif ('related' in definition and 'to' in definition):
            continue
            
        # Context detection
        
        # Let there be two candidate adjectives, A and B. And we want to check that B is intensified word of A. 
        # In this case, as above, I just checked weather there is B in the definition of A
        # But there can be chances that B is not a adjective that has meaning of A, but just a adjective that
        # decorates some nouns in the definition.
        # So I suppose that B cannot be a pair of A when B is used to decorate some nouns in the definition
        
        # And there there is also case tha adjective and noun have same form, that B is noun but detected by
        # algorithm. I detected this case by checking 'a', 'the' ahaed of the word, then took out those cases
        if (c in definition):
            if ((len(definition) == 1)):
                if (find_adverbs(definition, c) != None):
                    adjectives_pair.append([w_name, c, find_adverbs(definition, c)])
                continue

            elif (len(definition) == 2):
                if (definition[1] in all_nouns_name):
                    continue
                elif (definition[1] in all_nouns_name):
                    continue
                elif (definition[1] in all_nouns_name):
                    continue
                if (find_adverbs(definition, c) != None):
                    adjectives_pair.append([w_name, c, find_adverbs(definition, c)])
                continue                
                
            for prev, cur, next in ngrams(definition, 3):
                if (c != cur):
                    continue

                if (prev == 'a' or prev == 'the'):
                    continue

                if (next in all_nouns_name):
                    continue
                elif (next[:-1] in all_nouns_name):
                    continue
                elif (next[:-2] in all_nouns_name):
                    continue

                if (find_adverbs(definition, c) != None):
                    adjectives_pair.append([w_name, c, find_adverbs(definition, c)])
                break
            
# adjectives_pair = list(set(map(tuple, adjectives_pair)))

# Remove duplicates with the contents order remained
new_list = []
for v in adjectives_pair:
    if v not in new_list:
        new_list.append(v)
        
adjectives_pair = new_list


# In[4]:


verbs_pair = []

for w in all_verbs:
    w_name = w.name().split('.')[0]
    for c in all_verbs_name:
        
        # Negate word detection
        # This kind of words change the meaning, so I took out those cases
        if (w_name == c):
            continue
            
        definition = w.definition().split()
        
        # Word detection
        if ('no' in definition):
            continue
        elif ('not' in definition):
            continue
        elif ('without' in definition):
            continue
        elif ('lacking' in definition):
            continue
            
        # Unnecessary word detection
        # some words are both verb and adverb, especillay frequent in most of definitions
        # so I took out those cases
        elif ('more' in definition):
            continue
        elif ('of' in definition):
            continue
        elif ('in' in definition):
            continue
        elif ('up' in definition):
            continue
        elif ('on' in definition):
            continue
        elif ('away' in definition):
            continue
        elif ('as' in definition):
            continue
        elif ('out' in definition):
            continue
        elif ('related' in definition and 'to' in definition):
            continue
            
        # Context detection
        
        # Let there be two candidate verbs, A and B. And we want to check that B is intensified word of A. 
        # In this case, as above, I just checked weather there is B in the definition of A
        # But there can be chances that B is not a adjective that has meaning of A, but just a adjective that
        # decorates some nouns in the definition. (there's word that verb and adverb have same form)
        # So I suppose that B cannot be a pair of A when B is used to decorate some nouns in the definition 
        
        # And there there is also case tha verb and noun have same form, that B is noun but detected by
        # algorithm. I detected this case by checking 'a', 'the' ahaed of the word, then took out those cases
        if (c in definition):
            if (len(definition) == 1):
                if (find_adverbs(definition, c) != None):
                    verbs_pair.append([w_name, c, find_adverbs(definition, c)])
                continue
                
            elif (len(definition) == 2):
                if (definition[1] in all_nouns_name):
                    continue
                elif (definition[1] in all_nouns_name):
                    continue
                elif (definition[1] in all_nouns_name):
                    continue 
                if (find_adverbs(definition, c) != None):
                    verbs_pair.append([w_name, c, find_adverbs(definition, c)])
                continue

            for prev, cur, next in ngrams(definition, 3):
                if (c != cur):
                    continue

                if (prev == 'a' or prev == 'the'):
                    continue

                if (next in all_nouns_name):
                    continue
                elif (next[:-1] in all_nouns_name and next.endswith('s')):
                    continue
                elif (next[:-2] in all_nouns_name and next.endswith('es')):
                    continue

                if (find_adverbs(definition, c) != None):
                    verbs_pair.append([w_name, c, find_adverbs(definition, c)])
                break
            
verbs_pair = list(set(map(tuple, verbs_pair)))

# Remove duplicates with the contents order remained
new_list = []
for v in verbs_pair:
    if v not in new_list:
        new_list.append(v)
        
verbs_pair = new_list


# In[5]:


result = adjectives_pair[:25] + verbs_pair[:25] + adjectives_pair[25:] + verbs_pair[25:]

data = result[:50]

csv_file = open("./CS372_HW1_output_20160009.csv", "w", newline="")

csv_writer = csv.writer(csv_file)

for row in data:
    csv_writer.writerow(row)
    
csv_file.close()


# In[ ]:




