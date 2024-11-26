import re
from collections import Counter

def find_frequent_words(file_path, n):
    try:
        with open(file_path, 'r') as file:
            text = file.read()
    except FileNotFoundError:
        print("File not found!")
        return []
    
    words = re.findall(r'\b\w+\b', text.lower())
    
    word_counts = Counter(words)
    
    most_common_words = word_counts.most_common(n)
    
    result = [word for word, count in most_common_words]
    
    return result

file_path = "shakespeare.txt"
n = 5
frequent_words = find_frequent_words(file_path, n)

print(f"The {n} most frequent words are:")
for word in frequent_words:
    print(word)
