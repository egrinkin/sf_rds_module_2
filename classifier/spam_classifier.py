import re
import math
import shelve
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

stop_words=list(ENGLISH_STOP_WORDS)
pattern = re.compile('\w+')

class Classifier:
    
    spam_words = {}
    not_spam_words = {}
    spam_words_new = {}
    not_spam_words_new = {}
    s = shelve.open("train_data")
    
    def calculate_word_frequencies(self, body, label):
        """Заполняет словари спам-слов и неспам-слов: ключ - слово, 
        значение - количество раз, когда это слово встречется в письмах"""
        # Функция используется как для обучения классификатора (label='SPAM' или label='NOT_SPAM'), 
        # так и для заполнения словарей незнакомыми словами при классификации (label=None)
        # Обучение
        if label == 'SPAM':
            for word in body:                                          
                if word not in Classifier.spam_words.keys():
                    Classifier.spam_words[word] = 1
                    Classifier.not_spam_words[word] = 0
                else:
                    Classifier.spam_words[word] += 1
        elif label == 'NOT_SPAM':
            for word in body:   
                if word not in Classifier.not_spam_words.keys():
                    Classifier.not_spam_words[word] = 1
                    Classifier.spam_words[word] = 0
                else:
                    Classifier.not_spam_words[word] += 1
        # Классификация
        elif label == None:                         
            spam_words = Classifier.s['spam_words']
            not_spam_words = Classifier.s['not_spam_words']
            for word in body: 
                if word not in spam_words.keys():
                    spam_words[word] = 0
                    not_spam_words[word] = 0 
            Classifier.spam_words_new = spam_words
            Classifier.not_spam_words_new = not_spam_words

    def del_stop_words(self, cell):
        new_cell = []
        for word in cell:
            if word not in stop_words:
                new_cell.append(word)
        return new_cell
      
    def train(self):
        """Обучает классификатор: заполняет словари спам-слов и неспам-слов, 
        подсчитывает значения переменных pA и pNotA."""
        df=pd.read_csv('spam_or_not_spam.csv')
        df=df.dropna()
        df['label']=df['label'].replace([1, 0], ['SPAM', 'NOT_SPAM'])
        df['email']=df['email'].apply(lambda x: list(map(lambda y: y.lower(), pattern.findall(x))))
        df['email']=df['email'].apply(self.del_stop_words)
        Classifier.spam_words.clear()
        Classifier.not_spam_words.clear()
        spam_letters_number = 0
        total_letters_number = 0
        for i in range(df.shape[0]):
            self.calculate_word_frequencies(df['email'].iloc[i], df['label'].iloc[i])
            if df['label'].iloc[i] == 'SPAM':
                spam_letters_number += 1
                total_letters_number += 1
            else:
                total_letters_number += 1
        pA = spam_letters_number/total_letters_number
        pNotA = 1 - pA 
        Classifier.s['pA'] = pA
        Classifier.s['pNotA'] = pNotA
        Classifier.s['spam_words'] = Classifier.spam_words
        Classifier.s['not_spam_words'] = Classifier.not_spam_words
        

    def calculate_P_Bi_A(self, word, label):   
        """Считает для каждого слова натуральные логарифмы вероятностей того, что это спам, и того, что это не спам"""
        if label == 'SPAM':
            return math.log((Classifier.spam_words_new[word] + 1)/(len(Classifier.spam_words_new) + (sum(Classifier.spam_words_new.values()))))
        if label == 'NOT_SPAM': 
            return math.log((Classifier.not_spam_words_new[word] + 1)/(len(Classifier.not_spam_words_new) + sum(Classifier.not_spam_words_new.values())))

    def calculate_P_B_A(self, text, label):
        """Подсчитывает натуральный логарифм вероятности спама и неспама для строки"""
        if label == 'SPAM':
            P_B_A = 0
            for word in text:
                P_B_A += self.calculate_P_Bi_A(word, 'SPAM')
            return P_B_A
        if label == 'NOT_SPAM':
            P_B_NotA = 0
            for word in text:
                P_B_NotA += self.calculate_P_Bi_A(word, 'NOT_SPAM')
            return P_B_NotA 

    def classify(self, email):
        """Классифицирует письмо как спам или неспам"""
        pA = Classifier.s['pA']
        pNotA = Classifier.s['pNotA']
        email = list(map(lambda x: x.lower(), pattern.findall(email)))
        email = self.del_stop_words(email)
        self.calculate_word_frequencies(email, label = None)
        P_B_A = self.calculate_P_B_A(email, 'SPAM')
        P_B_NotA = self.calculate_P_B_A(email, 'NOT_SPAM')
        if math.log(pA) + P_B_A > math.log(pNotA) + P_B_NotA:
            return 'CПАМ!'
        else:
            return 'НЕ СПАМ!' 
