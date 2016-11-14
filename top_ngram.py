from collections import Counter
import json
import nltk

symbols = [" ", "\'", "\"", "-"]

def main():
    headlines = load_headlines()
    result = []
    for line in headlines:
        result.append(get_ngrams(line))
    # print result
    results_flattened = [item for sublist in result for item in sublist]
    # print results_flattened
    ngrams = Counter(results_flattened)
    # print len(ngrams)
    final_result = ngrams.most_common(200)
    print final_result

    ultimate = [key for key,value in final_result]
    result =  [" ".join(ng) for ng in ultimate]
    with open('top_ngrams.json', 'w+') as outfile:
        outfile.write("\n".join(result))

def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])

def get_ngrams(line):
    result = []
    toker = nltk.RegexpTokenizer(r'((?<=[^\w\s])\w(?=[^\w\s])|(\W))+', gaps=True)
    words_ = toker.tokenize(line)
    # words_ = nltk.word_tokenize(line)

    words = [token for token in words_ if token not in symbols]
    # print words
    for i in range(3, 6):
        result = result + find_ngrams(words, i)
    return result


def load_headlines():
    datafile = open('clickBaitData.json', 'r')
    return [json.loads(line)['title'] for line in datafile]

if __name__ == '__main__':
    main()
