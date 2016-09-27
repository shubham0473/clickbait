import nltk
import json
import nltk


def main():
    headlines = load_headlines()
    result = []
    for line in headlines:
        result.append({line: get_no_of_tokens(line)})
    with open('sent_len_feature.json', 'w+') as outfile:
        json.dump(result, outfile, indent=0)

    # print data

def get_no_of_tokens(line):
    tokens = nltk.word_tokenize(line)
    return len(tokens)
def load_headlines():
    datafile = open('clickBaitData.json', 'r')
    return [json.loads(line)['title'] for line in datafile]

if __name__ == '__main__':
    main()
