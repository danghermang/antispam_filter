import os
import pickle
import re
from typing import Counter
import mailparser
from nltk import PorterStemmer, LancasterStemmer
from sklearn.ensemble import AdaBoostClassifier
from scipy.special import log1p
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

clean_paths = [r'.\Lot1\Clean',
               r'.\Lot2\Clean',
               r'.\Lot3\Clean'
               ][:2]
spam_paths = [r'.\Lot1\Spam',
              r'.\Lot2\Spam',
              r'.\Lot3\Spam'
              ][:2]
all_paths = [r'.\Lot1\Clean',
             r'.\Lot2\Clean',
             r'.\Lot1\Spam',
             r'.\Lot2\Spam',
             # r'.\Lot3\Clean',
             # r'.\Lot3\Spam'
             ]


def get_words_from_string(string):
    string = string.lower()
    word_pattern = r'[A-Za-z]+'
    # link_pattern = r"(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9]\.[^\s]{2,})"
    # email_pattern = r"\S+@\S+"
    # ip_pattern = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    result = []
    # for x in re.findall(link_pattern, string):
    #     try:
    #         url = "{0.scheme}://{0.netloc}/".format(urlsplit(x))
    #     except:
    #         url = x
    #     result.append(url)
    # string = re.sub(link_pattern, "", string)
    # result.extend(re.findall(email_pattern, string))
    # string = re.sub(email_pattern, "", string)
    # result.extend(re.findall(ip_pattern, string))
    # string = re.sub(ip_pattern, "", string)
    # stemmer = PorterStemmer()
    stemmer = LancasterStemmer()
    result.extend([stemmer.stem(word) for word in re.findall(word_pattern, string)])
    # result.extend(re.findall(word_pattern, string))
    return result
    # stemmer = EnglishStemmer()
    # return stemmer.stemWords(re.findall(word_pattern, string))


def get_string_from_words(words):
    return "".join([x + " " for x in words])


def parse_mail(path):
    read_mail = mailparser.parse_from_file(path)
    body = read_mail.body
    return get_words_from_string(body)


def sterge_cacat():
    files = [os.path.join(r'.\Lot3\Spam', file_name) for file_name in os.listdir(r'.\Lot3\Spam')]
    files.extend([os.path.join(r'.\Lot3\Clean', file_name) for file_name in os.listdir(r'.\Lot3\Clean')])
    for fisier in files:
        try:
            if len(parse_mail(fisier)) == 0:
                raise Exception
        except Exception as e:
            os.remove(fisier)
            print(e, "\nsters", fisier)


def make_Dictionary(folder_paths):
    mail_paths = []
    for folder_path in folder_paths:
        mail_paths.extend([os.path.join(folder_path, file_name) for file_name in os.listdir(folder_path)])
    all_words = []
    for mail_path in mail_paths:
        print("parsing", mail_path)
        all_words += parse_mail(mail_path)

    dictionary = Counter(all_words)
    return dictionary


def make_sentence(folder_paths):
    mail_paths = []
    for folder_path in folder_paths:
        mail_paths.extend([os.path.join(folder_path, file_name) for file_name in os.listdir(folder_path)])
    # vectorizer = TfidfVectorizer(norm='l2',min_df=0.02,max_df=0.80,ngram_range=(1,4))
    vectorizer = TfidfVectorizer(norm='l2')
    all_sentences = []
    all_labels = []
    for mail_path in mail_paths:
        print("parsing", mail_path)
        all_sentences.append(get_string_from_words(parse_mail(mail_path)))
        if "clean" in mail_path.lower():
            all_labels.append(1)
        else:
            all_labels.append(-1)
    return vectorizer.fit(all_sentences), vectorizer.fit_transform(all_sentences), all_labels


def get_score(words, only_clean, only_spam):
    score = 0
    if len(words) == 0:
        return -1
    for word in words:
        if word in only_clean:
            score += only_clean[word]
        elif word in only_spam:
            score -= only_spam[word]
        if score > 10 or score < -10:
            break
    return score


def get_better_dicts(clean, spam):
    only_clean_json = {}
    only_spam_json = {}
    all_keys = (clean | spam).keys()
    # max_value = max(max([clean[key] for key in clean]),max(spam[key] for key in spam))
    for key in all_keys:
        if len(key) <= 2:
            continue
        if key not in spam:
            only_clean_json[key] = log1p(clean[key])
        elif key not in clean:
            only_spam_json[key] = log1p(spam[key])
        # else:
        #     if clean[key] > spam[key]:
        #         only_clean_json[key] = expit((clean[key] - spam[key])/10)
        #     elif spam[key] > clean[key]:
        #         only_spam_json[key] = expit((spam[key] - clean[key])/10)
    return only_clean_json, only_spam_json


def get_representation(mail, vectorizer):
    string = get_string_from_words(parse_mail(mail))
    return vectorizer.transform([string])


def get_representation_list(mails, vectorizer):
    strings = [get_string_from_words(parse_mail(x)) for x in mails]
    return vectorizer.transform(strings)


if __name__ == "__main__":
    all_vectorizer, all_transformed, all_labels = make_sentence(all_paths)

    with open('all_vectorizer.pkl', 'wb') as f:
        pickle.dump(all_vectorizer, f)
    with open('all_transformed.pkl', 'wb') as f:
        pickle.dump(all_transformed, f)
    with open('all_labels.pkl', 'wb') as f:
        pickle.dump(all_labels, f)

    with open('all_vectorizer.pkl', 'rb') as f:
        all_vectorizer = pickle.load(f)
    with open('all_transformed.pkl', 'rb') as f:
        all_transformed = pickle.load(f)
    with open('all_labels.pkl', 'rb') as f:
        all_labels = pickle.load(f)

    # model = MultinomialNB().fit(all_transformed, all_labels)
    model = AdaBoostClassifier().fit(all_transformed,all_labels)
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    wrong = []
    for path in clean_paths + spam_paths:
        counter = 0
        files = [os.path.join(path, mail_file) for mail_file in os.listdir(path)]
        repr = get_representation_list(files, all_vectorizer)
        scores = model.predict(repr)
        for file, score in zip(files, scores):
            if (score >= 0 and "clean" in file.lower()) or (score < 0 and "spam" in file.lower()):
                counter += 1
            else:
                wrong.append([parse_mail(file),score])
        print("Got", (counter / len(files)) * 100, "acc on", path)

    # number_of = 3000
    # clean_json = make_Dictionary(clean_paths)
    # spam_json = make_Dictionary(spam_paths)
    # with open('clean.json', 'w') as f:
    #     json.dump(clean_json, f)
    # with open('spam.json', 'w') as f:
    #     json.dump(spam_json, f)

    # with open('clean.json', 'r') as f:
    #     clean_json = Counter(json.load(f))
    # with open('spam.json', 'r') as f:
    #     spam_json = Counter(json.load(f))
    #
    # only_clean_json, only_spam_json = get_better_dicts(clean_json, spam_json)
    # with open('only_clean.json', 'w') as f:
    #     json.dump(only_clean_json, f)
    # with open('only_spam.json', 'w') as f:
    #     json.dump(only_spam_json, f)
    #
    # with open('only_clean.json', 'r') as f:
    #     only_clean_json = Counter(json.load(f))
    # with open('only_spam.json', 'r') as f:
    #     only_spam_json = Counter(json.load(f))

    # most_common_clean = only_clean_json.most_common(int(len(only_clean_json.keys())/1000))
    # most_common_spam = only_spam_json.most_common(int(len(only_spam_json.keys()) / 2))
    # only_clean_json = {only_clean_json[key] for key in most_common_clean}
    # only_spam_json = {only_spam_json[key] for key in most_common_spam}
    # wrong = []
    # for path in clean_paths + spam_paths:
    #     counter = 0
    #     files = [os.path.join(path, mail_file) for mail_file in os.listdir(path)]
    #     for fisier in files:
    #         document = parse_mail(fisier)
    #         score = get_score(document, only_clean_json, only_spam_json)
    #         if (score >= 0 and "clean" in path.lower()) or (score < 0 and "spam" in path.lower()):
    #             counter += 1
    #         else:
    #             wrong.append((fisier, score, document))
    #     print("Got", (counter / len(files)) * 100, "acc on", path)
    # with open("wrong.txt", 'w') as f:
    #     json.dump(wrong, f)
    # with open("wrong.txt", 'r') as f:
    #     wrong = json.load(f)
