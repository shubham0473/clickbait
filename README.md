### Features extracted (for both cb and non_cb): 
1. `get_no_of_tokens()` : Number of words in the headline
2. `get_avg_char_count()` : Average number of characters in words (of headline)
3. `sd_len()`: Maximum length of syntactic dependency between governing & dependent words. 
If there exists one syntactic dependency of length 2 and two of length 4, the output vector for a given headline would be **[0, 0, 1, 0, 2]**      i.e the value at the corresponding index would be 1 (number of such dependencies). The maximum number of tokens in any headline is 19, so a list with 19 elements has been created.
4. `sub_count()`: Get a list of all words used as subjects in a title
