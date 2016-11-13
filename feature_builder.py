import nltk
import json
import nltk
from nltk.tag import StanfordNERTagger

st = StanfordNERTagger("/opt/stanford-ner-2015-12-09/classifiers/english.muc.7class.distsim.crf.ser.gz" , "/opt/stanford-ner-2015-12-09/stanford-ner.jar", encoding='utf-8')
TAGS = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$'
, 'RB', 'RBR', 'RBS', 'RP', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']

def main():
    headlines = load_headlines()
    result = []
    for line in headlines:
        get_pos_stats(line)
        result.append({
            line: {
                'noOfTokens' : get_no_of_tokens(line),
                'avgCharCount' : get_avg_char_count(line),
                # 'noOfSlangs' : get_no_of_slangs(line),
                'wordNGrams' : get_ngrams(line),
                'posStats' : get_pos_stats(line),
                }
            })
    with open('sent_len_feature.json', 'w+') as outfile:
        json.dump(result, outfile, indent=4)

    # print data


def get_pos_stats(line):
    words = nltk.word_tokenize(line)
    classified_text = st.tag(words)

    lower_str = []
    for i in classified_text:
        # print i
        if i[1] != 'O':
            lower_str.append(i[0])
        else:
            lower_str.append(i[0].lower())
    tagged_words = nltk.pos_tag(lower_str)
    vector = {}
    for tag in TAGS:
        vector[tag] = 0
    for w in tagged_words:
        try:
            vector[w[1]] += 1
        except:
            pass
    return vector

def get_no_of_tokens(line):
    tokens = nltk.word_tokenize(line)
    return len(tokens)

def get_avg_char_count(line):
    tokens = tokens = nltk.word_tokenize(line)
    sum = 0
    for token in tokens:
        sum = sum + len(token)
    return sum/(1.0*len(tokens))

def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])

def get_ngrams(line):
    result = []
    words = nltk.word_tokenize(line)
    for i in range(1, 5):
        result.append(find_ngrams(words, i))
    return str(result)

def load_headlines():
    datafile = open('cb.json', 'r')
    return [json.loads(line)['title'] for line in datafile]

if __name__ == '__main__':
    main()
