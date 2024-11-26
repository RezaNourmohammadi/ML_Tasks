#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>

#define MAX_WORD_LEN 100
#define INITIAL_WORDS 1000 

typedef struct {
    char word[MAX_WORD_LEN];
    int count;
} WordFreq;

void to_lowercase(char *str) {
    for (int i = 0; str[i]; i++) {
        str[i] = tolower((unsigned char)str[i]);
    }
}

void process_line(const char *line, WordFreq **word_list, int *word_count, int *capacity) {
    char word[MAX_WORD_LEN];
    int word_index = 0;

    for (int i = 0; line[i] != '\0'; i++) {
        if (isalnum(line[i])) {
            word[word_index++] = line[i];
        } else if (word_index > 0) {
            word[word_index] = '\0';
            to_lowercase(word);
            int found = 0;

            for (int j = 0; j < *word_count; j++) {
                if (strcmp((*word_list)[j].word, word) == 0) {
                    (*word_list)[j].count++;
                    found = 1;
                    break;
                }
            }

            if (!found) {
                if (*word_count >= *capacity) {
                    *capacity *= 2; 
                    *word_list = realloc(*word_list, *capacity * sizeof(WordFreq));
                    if (*word_list == NULL) {
                        fprintf(stderr, "Memory allocation failed for word list\n");
                        exit(1); 
                    }
                }
                strcpy((*word_list)[*word_count].word, word);
                (*word_list)[*word_count].count = 1;
                (*word_count)++;
            }

            word_index = 0;
        }
    }
}

int compare_word_freq(const void *a, const void *b) {
    return ((WordFreq *)b)->count - ((WordFreq *)a)->count;
}

char **find_frequent_words(const char *path, int32_t n) {
    FILE *file = fopen(path, "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file\n");
        return NULL;
    }

    int capacity = INITIAL_WORDS;
    WordFreq *word_list = (WordFreq *)malloc(capacity * sizeof(WordFreq));
    int word_count = 0;

    char line[1024]; 
    while (fgets(line, sizeof(line), file)) {
        process_line(line, &word_list, &word_count, &capacity);
    }

    fclose(file);

    qsort(word_list, word_count, sizeof(WordFreq), compare_word_freq);

    if (n > word_count) {
        n = word_count;
    }

    char **result = (char **)malloc(n * sizeof(char *));
    if (result == NULL) {
        fprintf(stderr, "Memory allocation failed for result array\n");
        return NULL; 
    }

    for (int i = 0; i < n; i++) {
        result[i] = strdup(word_list[i].word); 
        if (result[i] == NULL) {
            fprintf(stderr, "Memory allocation failed for word: %s\n", word_list[i].word);
            for (int j = 0; j < i; j++) {
                free(result[j]); 
            }
            free(result);
            free(word_list);
            return NULL;  
        }
    }

    free(word_list);

    return result;
}

int main() {
    const char *file_path = "shakespeare.txt";
    int n = 5;

    char **frequent_words = find_frequent_words(file_path, n);

    if (frequent_words != NULL) {
        printf("The %d most frequent words are:\n", n);
        for (int i = 0; i < n; i++) {
            printf("%s\n", frequent_words[i]);
            free(frequent_words[i]); 
        }
        free(frequent_words);
    }

    return 0;
}
