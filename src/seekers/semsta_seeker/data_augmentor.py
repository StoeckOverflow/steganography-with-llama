import json
import os
import nltk
from nltk.corpus import wordnet
import numpy as np
import random
from transformers import pipeline
from llama_cpp import Llama
#nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger')
translate_to_de = pipeline("translation_en_to_de", model="Helsinki-NLP/opus-mt-en-de")
translate_to_en = pipeline("translation_de_to_en", model="Helsinki-NLP/opus-mt-de-en")

class DataAugmentor:
    def __init__(self):
        self.llm = Llama(model_path="llama-2-7b.Q5_K_M.gguf", seed=1337, verbose=False, logits_all=True, n_threads=None, use_mlock=True)
    
    def synonym_replacement(self, sentence, n):
        words = nltk.word_tokenize(sentence)
        new_words = words.copy()
        random_word_list = list(set([word for word in words if word.isalnum()]))
        num_replaced = 0
        for random_word in random_word_list:
            synonyms = self.get_synonyms(random_word)
            if len(synonyms) >= 1:
                synonym = random.choice(list(synonyms))
                new_words = [synonym if word == random_word else word for word in new_words]
                num_replaced += 1
            if num_replaced >= n: # n is the number of words you want to replace
                break

        sentence = ' '.join(new_words)
        return sentence
    
    def synonym_replacement_perturb_k_words(self, sentence, k=3):
        words = nltk.word_tokenize(sentence)
        new_words = []
        for i in range(len(words)):
            word = words[i]
            # Replace every third word with a synonym, if it is alphanumeric and synonyms exist
            if (i + 1) % k == 0 and word.isalnum():
                synonyms = self.get_synonyms(word)
                if len(synonyms) >= 1:
                    synonym = random.choice(synonyms)
                    new_words.append(synonym)
                else:
                    # If no synonym exists, keep the original word
                    new_words.append(word)
            else:
                # Keep all other words as they are
                new_words.append(word)

        sentence = ' '.join(new_words)
        return sentence

    def get_synonyms(self, word):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace("_", " ").replace("-", " ").lower()
                synonym = "".join([char for char in synonym if char in ' abcdefghijklmnopqrstuvwxyz'])
                synonyms.add(synonym)
        if word in synonyms:
            synonyms.remove(word)
        return list(synonyms)
    
    def back_translate(self, text, translation_pipelines=[translate_to_de, translate_to_en]):
        translated = translation_pipelines[0](text, max_length=100)[0]['translation_text']
        back_translated = translation_pipelines[1](translated, max_length=100)[0]['translation_text']
        
        return back_translated
    
    @staticmethod
    def resample_with_replacement(texts, labels):
        """
        Resample the given texts and labels with replacement.
        
        :param texts: List or array of text data.
        :param labels: Corresponding labels.
        :return: Resampled texts and labels.
        """
        n_samples = len(texts)
        indices = np.random.choice(np.arange(n_samples), size=n_samples, replace=True)
        resampled_texts = [texts[i] for i in indices]
        resampled_labels = [labels[i] for i in indices]
        return resampled_texts, resampled_labels

    def load_json_file(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data

    def save_json_file(self, data, filepath):
        with open(filepath, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
    
    def augment_article(self, article, n_synonyms=20):
        back_translated_article = self.back_translate(article)
        augmented_article = self.synonym_replacement(back_translated_article, n_synonyms)
        return augmented_article
    
    def perturb_feed_with_synonyms(self, feed):
        perturbed_feed = []
        for article in feed:
            perturbed_article = self.synonym_replacement_perturb_k_words(article, random.randint(3,6))
            perturbed_feed.append(perturbed_article)
        return perturbed_feed
    
    def perturb_feed_with_llama(self, feed):
        perturbed_feed = []
        max_words = 90
        max_tokens_estimate = max_words * 1.5
        for article in feed:
            prompt = article[:5]
            prompt_tokens = self.llm.tokenizer().encode(prompt)
            num_prompt_tokens = len(prompt_tokens)
            max_tokens_for_completion = max_tokens_estimate - num_prompt_tokens
            completion = self.llm(
                prompt=prompt,
                max_tokens=max_tokens_for_completion,
                temperature=0.7,
                top_p=0.9,
            )['choices'][0]['text']
            perturbed_article = prompt + completion
            perturbed_feed.append(perturbed_article)
        return perturbed_feed
    
    def strip_after_last_period(self, s):
        # Find the last occurrence of the period
        last_period_index = s.rfind('.')
        # If a period is found, return the substring up to the period
        if last_period_index != -1:
            return s[:last_period_index]
        else:
            # Return the original string if there's no period
            return s
    
    def create_augmented_datasets(self, directory_path, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        file_names = os.listdir(directory_path)
        feed_counter = 0

        num_feeds = len(file_names)
        all_indices = list(range(num_feeds))
        random.shuffle(all_indices)
        llama_indices = set(all_indices[:num_feeds // 4])  # 25% for LLaMA perturbation
        synonym_indices = set(all_indices[num_feeds // 4: num_feeds // 2])  # Another 25% for synonyms

        for i, file_name in enumerate(file_names):
            file_path = os.path.join(directory_path, file_name)
            data = self.load_json_file(file_path)

            original_feed = data['feed']
            augmented_feed = []
            original_feed = [self.back_translate(article) for article in original_feed]

            if i in llama_indices:
                # Perturb the first 4 articles with LLaMA
                for article in original_feed[:4]:
                    augmented_feed.append(self.perturb_feed_with_llama([article])[0])  # Assuming it returns a list of articles
                augmented_feed.extend(original_feed[4:])  # Add the rest of the articles unchanged
                label = -1
            elif i in synonym_indices:
                # Perturb the first 4 articles with synonym replacement
                for article in original_feed[:4]:
                    augmented_feed.append(self.synonym_replacement_perturb_k_words(article, random.randint(3, 6)))
                augmented_feed.extend(original_feed[4:])  # Add the rest of the articles unchanged
                label = -1
            else:
                # No perturbation
                augmented_feed = original_feed
                label = 1

            augmented_data = {'feed': augmented_feed}
            feed_counter += 1
            augmented_file_name = f"augmented_feed_{feed_counter:03d}.json;{label}"
            augmented_file_path = os.path.join(save_dir, augmented_file_name)
            self.save_json_file(augmented_data, augmented_file_path)

            print(f"Augmented dataset saved to {augmented_file_path}")