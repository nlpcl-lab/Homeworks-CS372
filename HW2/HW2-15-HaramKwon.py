import nltk
import csv
from nltk.corpus import brown
from nltk.sentiment.vader import SentimentIntensityAnalyzer as sia
'''
Step 1. Brown Corpus 의 tagged text를 받아온다.

Step 2. 다음과 같은 방법으로 Intensifier를 찾는다.
    2-1. tagged text에서 모든 adverb와 adjective를 찾는다.
    2-2. 2-1에서 찾은 adjective와 adverb를 intensifier 후보로 삼는다.
        후보 intensifiers 의 뒤에 오는 word를 합해 pair를 만들고,
        1. 수식받는 word 의 sentiment intensity의 절대값이 0보다 클 때, 
        2. word의 intensity 절대값보다 intensifier까지 붙은 pair의 절대값이 더 크며,
        3. intensity의 절대값이 커지되 부호는 그대로인 경우
        intensify한다고 받아들여 해당 pair의 adjective 또는 adverb는 intensifiers로 분류한다.
    2-3. 2-2에서 분류한 pair는 intensifier + 수식받는 word 의 pair로 간주한다.
    
Step 3. 2-3에서 분류한 intensifier + 수식받은 word pair를 정리한다.
    intensifier별로 분류하고, intensifier마다 수식받은 word의 list와 각각의 occurrence 정보를 담은
    list를 얻어낸다. 이 때 얻어낸 list는 나중에 score를 계산하는데 이용된다.
    
Step 4. Score를 계산한다.
    Intensifier + 수식받는 word pair에 대한 Score 계산 방법은 다음과 같다.
    우선 앞서 구한 intensifier임에도 불구하고 step2에서 설정한 기준에 따라 intensifier가 쓰인 word pair 중
    step 2에서 걸러졌을 수도 있다. 우리는 거기서 걸러진 pair 중, intensifier가 적절한 pos (adj 또는 adverb)를 갖고
    뒤에 오는 단어를 수식하는 모든 pair가 있으면 그러한 pair까지 포함시킨 보다 방대한 intensifier + 수식받는 word pair의
    list를 만들 것이다. 
    다시 말해 우리는 list 2개가 있다.
    1. step 2의 기준을 통과한 intensifier + 수식받는 word의 list
    2. step 2에서 intensifier로 분류된 단어 + 수식받는 word 의 모든 pair
    
    score를 따질 때 특정 intensifier가 꾸며주는 단어의 종류수를 따져야 할 일이 있는데, 이 때
    list1은 intensifier가 꾸며주는 모든 단어를 나타내지 못하기 때문에 list2의 존재가 필요하다.
    
    (first_score, occurrence/total, occurrence) 가 score vector다.
    first_score의 크기가 큰 것을 최우선으로 따지며, 
    first_score이 동률일 경우 second score에 해당하는 occurence의 크기가 큰 것을 따지며,
    second_score까지 동률일 경우 occurrence/total이 높은 것을 따진다.
    first_score: 하나의 intensifier에 대해 수식받은 word의 종류 수를 기준으로 점수를 매기며,
                수식받은 word의 종류가 다양할수록 less unique하고 이용 범위가 다양해 less restrictive하다고
                 볼 수 있기 때문에 종류 가지수를 분모로 설정해 first_score를 계산한다.
                (e.g. text내에서 intensifier 'very'에 의해 수식된 단어의 가짓수가 40이라고 한다면, 
                      first_score의 값은 1/40이 된다.)
    occurrence/total: first_score이 같을 경우, uniqueness가 얼추 비슷하다고 보기 때문에 restrictiveness를 따진다.
                     first_score이 같아도 intensifier가 다른 단어들에 비해 특정 단어를 수식하는 비중이 높다면 해당 조합은
                    다른 단어들에 비해 더욱 많이 쓰여 restrictiveness가 다른 단어를 수식할 때에 비해 보다 높으며, 해당 조합의
                    사용이 어느 정도 범용적이라는 점에 있어 신뢰도가 높아진다.                      
    occurrence: first_score이 동률일 경우, 단순히 더욱 많이 쓰인 표현을 보다 unique한다고 본다. 그 이유는 다음과 같다.
                해당 표현이 특히 많이 쓰이는 데에는 다양한 이유가 있을 수 있지만, 그 이유들 중 하나로는 이를 대체할 표현이 한정적이기 
                때문에 많이 이용된 것일 수도 있다. 따라서 특정 intensifier + 수식받는 word 의 조합의 occurrence가 다른 것들에
                비해 많을수록 보다 unique한 표현이라고 생각해 점수 측정에 추가했다.

Step 5. Score sort
    Step 4에서 점수를 계산하고 무작위로 나열된 것을 점수에 따라 나열한다.
    점수는 (first_score, second_score, third_score)의 형태인데,
    우선적으로 first_score의 크기에 따라 내림차순으로 나열하고, 
    first_score이 동률이면 해당 그룹 안에서 second_score의 크기에 따라 내림차순으로 나열한다. 
    second_score이 동률인 경우에는 third_score의 크기 비교를 통해 점수의 우위를 가리며, 
    third_score이 높을수록 보다 높은 점수를 갖는 방식으로 sort한다.

Step 6. csv file로 정리.
    

'''

def intensifier_candidate(tagged_text):
    intensity_adv_index = []
    intensity_adj_index = []
    for i, pair in enumerate(tagged_text):
        if pair[1] == 'ADJ':
            intensity_adj_index.append(i)
        elif pair[1] == 'VERB':
            intensity_adv_index.append(i)
        else:
            continue

    return intensity_adv_index, intensity_adj_index


sign = lambda x: (x>0) - (x<0)
# intensifier 후보로부터 adverb인 intensifier 찾아낸다.
def adv_intensifier(tagged_text, adv_candid_index):
    intensifier_list = []
    pairlist = []
    sid = sia()
    for i in adv_candid_index:
        intensifier = tagged_text[i][0].strip('[\'\"]')
        word_one = tagged_text[i - 1][0].strip('[\'\"]')
        word_two = tagged_text[i + 1][0].strip('[\'\"]')
        pair_one = word_one + ' ' + intensifier
        pair_two = intensifier + ' ' + word_two
        word_intensity = sid.polarity_scores(word_two)['compound']
        pair_intensity = sid.polarity_scores(pair_two)['compound']
        # 아래 if의 조건을 만족할 경우 현재 loop에서 돌고 있는 adverb는 intensifier의 역할을 한다.
        if abs(word_intensity) > 0 and abs(word_intensity) < abs(pair_intensity) and\
            sign(word_intensity) == sign(pair_intensity):
            if tagged_text[i + 1] == 'VERB' or tagged_text[i + 1][1] == 'ADJ':
                intensifier_list.append(intensifier.lower())
                pairlist.append((intensifier.lower(), word_two.lower()))

    return intensifier_list, pairlist

# intensifier 후보로부터 adjective인 intensifier 찾아낸다.
def adj_intensifier(tagged_text, adj_candid_index):
    intensifier_list = []
    pairlist = []
    sid = sia()
    for i in adj_candid_index:
        intensifier = tagged_text[i][0].strip('[\'\"]')
        word= tagged_text[i + 1][0].strip('[\'\"]')
        pair = word + ' ' + intensifier
        word_intensity = sid.polarity_scores(word)['compound']
        pair_intensity = sid.polarity_scores(pair)['compound']
        # 아래 if의 조건을 만족할 경우 현재 loop에서 돌고 있는 adjective는 intensifier의 역할을 한다.
        if abs(word_intensity) > 0 and abs(word_intensity) < abs(pair_intensity) and\
                sign(word_intensity) == sign(pair_intensity):
            if tagged_text[i + 1][1] == 'NOUN':
                intensifier_list.append(intensifier.lower())
                pairlist.append((intensifier.lower(), word.lower()))

    return intensifier_list, pairlist

# intensifier가 adverb인 D, E pair를 text에서 찾는다.
def adv_pair(tagged_text, adv_intensifier_list):
    adv_pair_list = []
    for i, pair in enumerate(tagged_text):
        if (pair[0].lower() in adv_intensifier_list) and pair[1] == 'ADV' and (tagged_text[i + 1][1] == 'VERB' or tagged_text[i + 1][1] == 'ADJ'):
            adv_pair_list.append((pair[0].lower(), tagged_text[i + 1][0].lower()))
        else:
            continue

    return adv_pair_list

# intensifier가 adjective인 D, E pair를 text에서 찾는다.
def adj_pair(tagged_text, adj_intensifier_list):
    adj_pair_list = []
    for i, pair in enumerate(tagged_text):
        if (pair[0].lower() in adj_intensifier_list) and pair[1] == 'ADJ' and tagged_text[i + 1][1] == 'NOUN':
            adj_pair_list.append((pair[0].lower(), tagged_text[i + 1][0].lower()))
        else:
            continue

    return adj_pair_list

# D, E pair를 구한 뒤 추후에 score를 구하기 쉬운 형태로 정리한다.
def pair_list_organized(common_fd):
    '''
    From the frequency distribution of a pair of intensifier and intensified word,
    it seeks how frequent certain pair of intensified word and an intensifier are used.
    It is organized by used intensifier, and along the intensifier there are intensified
    words and number of occurrences tied together.

    :param common_fd: frequency distribution of a pair
    :return: a list organized in the form of [[intensifier, (intensified word, number of occurrences)*] *]
    * means multiple here
    '''
    proportion = []
    proportion_occurrence = []
    for pair in common_fd:
        modifier = pair[0][0]
        if modifier not in proportion_occurrence:
            proportion_occurrence.append(modifier)
            proportion.append([modifier])
        intensifier_index = proportion_occurrence.index(modifier)
        proportion[intensifier_index].append((pair[0][1], pair[1]))

    return proportion #비율 계산한 것도 반영할 것!

# 바로 위의 함수에 의해 정렬된 데이터로 pair별로 score를 구한다.
def restrictiveness_score(list_organized, intensifier_pairlist):
    score_list = []
    for intensifier in list_organized:
        first_score = 1/len(intensifier[1:]) # 얼마나 다양하게 쓰이는지 나타내는 score
        occurrence_list = [pair[1] for pair in intensifier[1:]]
        total = sum(occurrence_list) # 현재 살펴보는 intensifier가 사용된 총 횟수
        for pair in intensifier[1:]:
            if (intensifier[0], pair[0]) in intensifier_pairlist:
                occurrence = pair[1] # 해당 pair이 몇 번 사용됐는지
                score_list.append([(intensifier[0], pair[0]), (first_score, occurrence / total, occurrence)])
            else:
                continue

    return score_list

# score를 내림차순으로 sort한다.
def score_sort(score_list):
    sorted_score_list = sorted(score_list, key = lambda x: x[1])
    return sorted_score_list[::-1]



# Step 1. POS tagged text 받아오기
text_tagged = brown.tagged_words(tagset='universal')

# Step 2-1. 모든 adverb와 adjective 추려내기.
adv_candid_index, adj_candid_index = intensifier_candidate(text_tagged)

# Step 2-2, 2-3. Intensifier를 선정하고, intensifier을 걸러내는 기준을 만족한 (intensifier, word) pair를 받는다.
intensifier_adverb, adv_pairlist = adv_intensifier(text_tagged, adv_candid_index)
intensifier_adjective, adj_pairlist = adj_intensifier(text_tagged, adj_candid_index)
intensifier_pairlist = adv_pairlist + adj_pairlist

# Step 3. step 4를 위한 text내 '모든' (intensifier, word) pair를 걸러낸다.
adv_pair_list = adv_pair(text_tagged, intensifier_adverb)
adj_pair_list = adj_pair(text_tagged, intensifier_adjective)
pair_list = adv_pair_list + adj_pair_list

# step 4에서 계산하기 용이하게 (intensifier, word) pair를 정리한다.
intensifier_fd = nltk.FreqDist(pair_list).most_common()
intensifier_organized = pair_list_organized(intensifier_fd)

# Step 4. 주어진 data를 이용해 점수 계산을 하고 (intensifier, word) pair에 점수를 tagging한다.
score_list = restrictiveness_score(intensifier_organized, intensifier_pairlist)

# Step 5. 점수에 따라 내림차순으로 나열한다.
sorted_score = score_sort(score_list)
ranked_hundred = sorted_score[:100] # 100개의 ranked D, E set
initial_hundred = score_list[:100] # 100개의 initial D, E set

# Step 6. 내림차순으로 첫 100개의 pair를 정리함과 동시에 (score 계산 과정에서 임의로 정리된) initial expression도 함께 정렬한다.
with open('CS372_HW2_output_20150069.csv', 'w', newline = '') as file:
    writer = csv.writer(file)
    writer.writerow(['(D) Ranked_intensifier', '(E) Ranked_word modified',
                     '(D) Initial_intensifier', '(E) Initial_word_modified'])
    for i in range(100):
        writer.writerow([ranked_hundred[i][0][0], ranked_hundred[i][0][1],
                         initial_hundred[i][0][0], initial_hundred[i][0][1]])

