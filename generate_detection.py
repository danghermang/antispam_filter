import os
import pickle
import re
import mailparser
from keras import backend as K
from keras import models, layers, callbacks
from sklearn.feature_extraction.text import TfidfVectorizer

K.clear_session()


all_paths = [
    r'.\Lot1\Clean',
    r'.\Lot1\Spam',
    r'.\Lot2\Clean',
    r'.\Lot2\Spam',
]


def get_words_from_string(string):
    string = string.lower()
    word_pattern = r'[A-Za-z]+'
    return [x.lower() for x in re.findall(word_pattern,string)]


def get_string_from_words(words):
    return "".join([x + " " for x in words])


def parse_mail(path):
    read_mail = mailparser.parse_from_file(path)
    body = read_mail.body
    return get_words_from_string(body)


def make_sentence(folder_paths):
    mail_paths = []
    for folder_path in folder_paths:
        mail_paths.extend([os.path.join(folder_path, file_name) for file_name in os.listdir(folder_path)])
    vectorizer = TfidfVectorizer(norm='l2',min_df=0.005,max_df=0.80)
    all_sentences = []
    all_labels = []
    for mail_path in mail_paths:
        print("parsing", mail_path)
        all_sentences.append(get_string_from_words(parse_mail(mail_path)))
        if "clean" in mail_path.lower():
            all_labels.append(1)
        else:
            all_labels.append(0)
    return vectorizer.fit(all_sentences), vectorizer.fit_transform(all_sentences), all_labels


def get_model(input_size):
    model = models.Sequential()
    model.add(layers.Dense(8, activation='relu', input_shape=(input_size,),kernel_initializer="lecun_normal"))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(8, activation='relu',kernel_initializer="lecun_normal"))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(16, activation='relu',kernel_initializer="lecun_normal"))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(16, activation='relu',kernel_initializer="lecun_normal"))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(32, activation='relu', kernel_initializer="lecun_normal"))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(32, activation='relu', kernel_initializer="lecun_normal"))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(32, activation='relu', kernel_initializer="lecun_normal"))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation='sigmoid',kernel_initializer="lecun_normal"))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def get_representation(mail, vectorizer):
    string = get_string_from_words(parse_mail(mail))
    return vectorizer.transform([string])


def get_representation_list(mails, vectorizer):
    strings = [get_string_from_words(parse_mail(x)) for x in mails]
    return vectorizer.transform(strings)


if __name__ == "__main__":
    # all_vectorizer, all_transformed, all_labels = make_sentence(all_paths)
    #
    # with open('all_vectorizer.pkl', 'wb') as f:
    #     pickle.dump(all_vectorizer, f)
    # with open('all_transformed.pkl', 'wb') as f:
    #     pickle.dump(all_transformed, f)
    # with open('all_labels.pkl', 'wb') as f:
    #     pickle.dump(all_labels, f)

    with open('all_vectorizer.pkl', 'rb') as f:
        all_vectorizer = pickle.load(f)
    # with open('all_transformed.pkl', 'rb') as f:
    #     all_transformed = pickle.load(f)
    # print(all_transformed.shape)
    # raise Exception
    # with open('all_labels.pkl', 'rb') as f:
    #     all_labels = pickle.load(f)
    #
    # model = get_model(all_transformed.shape[1])
    # early_stop = callbacks.EarlyStopping('val_acc', patience=20, restore_best_weights=True)
    # history_rnn = model.fit(all_transformed, all_labels, epochs=300, batch_size=32,
    #                         callbacks=[early_stop],
    #                         shuffle=True, validation_split=0.25)
    #
    # with open('model.pkl', 'wb') as f:
    #     pickle.dump(model, f)

    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    wrong = []
    for path in all_paths:
        counter = 0
        files = [os.path.join(path, mail_file) for mail_file in os.listdir(path)]
        representation = get_representation_list(files, all_vectorizer)
        scores = model.predict_classes(representation)
        for file, score in zip(files, scores):
            if (score == 1 and "clean" in file.lower()) or (score == 0 and "spam" in file.lower()):
                counter += 1
            else:
                wrong.append([parse_mail(file), score])
        print("Got", (counter / len(files)) * 100, "acc on", path)
