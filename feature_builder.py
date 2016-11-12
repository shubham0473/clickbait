import nltk
import json

def main():
    headlines = load_headlines()
    result = []
    for line in headlines:
        result.append({line: {'noOfTokens' : get_no_of_tokens(line), 'avgCharCount' : get_avg_char_count(line),'noOfStopwords' : get_no_of_stopwords(line)}})
    with open('sent_len_feature.json', 'w+') as outfile:
        json.dump(result, outfile, indent=4)

    # print data

def get_no_of_tokens(line):
    tokens = nltk.word_tokenize(line)
    return len(tokens)

def get_avg_char_count(line):
    tokens = tokens = nltk.word_tokenize(line)
    sum = 0
    for token in tokens:
        sum = sum + len(token)
    return sum/(1.0*len(tokens))

def load_headlines():
    datafile = open('clickBaitData.json', 'r')
    return [json.loads(line)['title'] for line in datafile]

def get_no_of_stopwords(line):
    stop = nltk.corpus.stopwords.words('english')
    return len([i for i in line.lower().split() if i in stop])


if __name__ == '__main__':
    main()
