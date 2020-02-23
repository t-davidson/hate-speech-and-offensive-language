"""
This file contains code to
    (a) Train a model on tagged data
    (b) Load a pre-trained classifier and
    associated files.
    (c) Transform new input data into the
    correct format for the classifier.
    (d) Run the classifier on the transformed
    data and return results .
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.externals import joblib

import nltk
from nltk.stem.porter import *
import argparse
import pickle as pkl
import os

from nltk.sentiment.vader import SentimentIntensityAnalyzer as VS
from textstat.textstat import *


stopwords = nltk.corpus.stopwords.words("english")

other_exclusions = ["#ff", "ff", "rt"]
stopwords.extend(other_exclusions)

sentiment_analyzer = VS()

stemmer = PorterStemmer()

other_features_names = ["FKRA", "FRE", "num_syllables",
                        "avg_syl_per_word", "num_chars",
                        "num_chars_total", "num_terms",
                        "num_words", "num_unique_words",
                        "vader neg", "vader pos", "vader neu",
                        "vader compound", "num_hashtags",
                        "num_mentions", "num_urls", "is_retweet"]
other_features_map = {name: i for i, name in enumerate(other_features_names)}

FILE_PATH = "D:\\thesis\\other_code\\hate-speech-and-offensive-language\\classifier"


def preprocess(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE
    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                       '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', str(text_string))
    parsed_text = re.sub(giant_url_regex, 'URLHERE', parsed_text)
    parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)
    return parsed_text


def tokenize(tweet):
    """Removes punctuation & excess whitespace, sets to lowercase,
    and stems tweets. Returns a list of stemmed tokens."""
    tweet = " ".join(re.split("[^a-zA-Z]*", tweet.lower())).strip()
    tokens = [stemmer.stem(t) for t in tweet.split()]
    return tokens


def basic_tokenize(tweet):
    """Same as tokenize but without the stemming"""
    tweet = " ".join(re.split("[^a-zA-Z.,!?]*", tweet.lower())).strip()
    return tweet.split()


def get_pos_tags(tweets):
    """Takes a list of strings (tweets) and
    returns a list of strings of (POS tags).
    """
    tweet_tags = []
    for t in tweets:
        tokens = basic_tokenize(preprocess(t))
        tags = nltk.pos_tag(tokens)
        tag_list = [x[1] for x in tags]
        tag_str = " ".join(tag_list)
        tweet_tags.append(tag_str)
    return tweet_tags


def count_twitter_objs(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE
    4) hashtags with HASHTAGHERE
    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned.
    Returns counts of urls, mentions, and hashtags.
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    hashtag_regex = '#[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, 'URLHERE', parsed_text)
    parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)
    parsed_text = re.sub(hashtag_regex, 'HASHTAGHERE', parsed_text)
    return parsed_text.count('URLHERE'), parsed_text.count('MENTIONHERE'),\
           parsed_text.count('HASHTAGHERE')


def other_features(tweet, other_idx=None):
    """This function takes a string and returns a list of features.
    These include Sentiment scores, Text and Readability scores,
    as well as Twitter specific features"""
    # SENTIMENT
    sentiment = sentiment_analyzer.polarity_scores(tweet)

    words = preprocess(tweet)  # Get text only

    syllables = textstat.syllable_count(words)  # count syllables in words
    num_chars = sum(len(w) for w in words)  # num chars in words
    num_chars_total = len(tweet)
    num_terms = len(tweet.split())
    num_words = len(words.split())
    avg_syl = round(float((syllables + 0.001)) / float(num_words + 0.001), 4)
    num_unique_terms = len(set(words.split()))

    # Modified FK grade, where avg words per sentence is just num words/1 (hard coded from original code)
    FKRA = round(float(0.39 * float(num_words) / 1.0) + float(11.8 * avg_syl) - 15.59, 1)
    # Modified FRE score, where sentence fixed to 1 (hard coded from original code)
    FRE = round(206.835 - 1.015 * (float(num_words) / 1.0) - (84.6 * float(avg_syl)), 2)

    twitter_objs = count_twitter_objs(tweet)  # Count #, @, and http://
    retweet = 0
    if "rt" in words:
        retweet = 1
    features = [FKRA, FRE, syllables, avg_syl, num_chars, num_chars_total, num_terms, num_words,
                num_unique_terms, sentiment['neg'], sentiment['pos'], sentiment['neu'], sentiment['compound'],
                twitter_objs[2], twitter_objs[1],
                twitter_objs[0], retweet]
    if other_idx:
        return [features[i] for i in other_idx]
    return features


def get_oth_features(tweets, other_idxs=None):
    """Takes a list of tweets, generates features for
    each tweet, and returns a numpy array of tweet x features"""
    feats = []
    for t in tweets:
        feats.append(other_features(t, other_idxs))
    return np.array(feats)


def transform_inputs(tweets, tf_vectorizer, idf_vector, pos_vectorizer, other_idxs):
    """
    This function takes a list of tweets, along with used to
    transform the tweets into the format accepted by the model.
    Each tweet is decomposed into
    (a) An array of TF-IDF scores for a set of n-grams in the tweet.
    (b) An array of POS tag sequences in the tweet.
    (c) An array of features including sentiment, vocab, and readability.
    Returns a pandas dataframe where each row is the set of features
    for a tweet. The features are a subset selected using a Logistic
    Regression with L1-regularization on the training data.
    """
    tf_array = tf_vectorizer.fit_transform(tweets).toarray()
    tfidf_array = tf_array*idf_vector
    print("Built TF-IDF array")

    pos_tags = get_pos_tags(tweets)
    pos_array = pos_vectorizer.fit_transform(pos_tags).toarray()
    print("Built POS array")

    oth_array = get_oth_features(tweets, other_idxs)
    print("Built other feature array")

    M = np.concatenate([tfidf_array, pos_array, oth_array], axis=1)
    return pd.DataFrame(M)


def predict(X, model):
    """
    This function calls the predict function on
    the trained model to generated a predicted y
    value for each observation.
    """
    y_preds = model.predict(X)
    return y_preds


def class_to_name(class_label):
    """
    This function can be used to map a numeric
    feature name to a particular class.
    """
    if class_label == 0:
        return "Hate speech"
    elif class_label == 1:
        return "Offensive language"
    elif class_label == 2:
        return "Neither"
    else:
        return "No label"


def get_tweets_predictions(tweets, model, pkl_dir, perform_prints=True):
    tf_pkl = os.path.join(os.path.join(pkl_dir, 'final_tfidf.pkl'))
    tfidf_vec = joblib.load(tf_pkl)
    idf_pkl = os.path.join(os.path.join(pkl_dir, 'final_idf.pkl'))
    idf_vec = joblib.load(idf_pkl)
    pos_pkl = os.path.join(os.path.join(pkl_dir, 'final_pos.pkl'))
    pos_vec = joblib.load(pos_pkl)
    other_pkl = os.path.join(os.path.join(pkl_dir, 'other_idxs.pkl'))
    other_idxs = joblib.load(other_pkl)
    X = prepare_data_from_pkls(tweets, tfidf_vec, idf_vec, pos_vec, other_idxs)

    predicted_class = predict(X, model)

    return predicted_class


def prepare_data_for_training(tweets):
    print(len(tweets), " tweets to classify")
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        preprocessor=preprocess,
        ngram_range=(1, 3),
        stop_words=stopwords,  # We do better when we keep stopwords
        use_idf=True,
        smooth_idf=False,
        norm=None,  # Applies l2 norm smoothing
        decode_error='replace',
        max_features=10000,
        min_df=5,
        max_df=0.501
    )
    # Construct tfidf matrix and get relevant scores
    tfidf = vectorizer.fit_transform(tweets).toarray()
    vocab = {v: i for i, v in enumerate(vectorizer.get_feature_names())}
    idf_vals = vectorizer.idf_
    # Get POS tags for tweets and save as a string
    tweet_tags = []
    for t in tweets:
        tokens = basic_tokenize(preprocess(t))
        tags = nltk.pos_tag(tokens)
        tag_list = [x[1] for x in tags]
        # for i in range(0, len(tokens)):
        tag_str = " ".join(tag_list)
        tweet_tags.append(tag_str)
        # print(tokens[i],tag_list[i])
    # We can use the TFIDF vectorizer to get a token matrix for the POS tags
    pos_vectorizer = TfidfVectorizer(
        tokenizer=None,
        lowercase=False,
        preprocessor=None,
        ngram_range=(1, 3),
        stop_words=None,  # We do better when we keep stopwords
        use_idf=False,
        smooth_idf=False,
        norm=None,  # Applies l2 norm smoothing
        decode_error='replace',
        max_features=5000,
        min_df=5,
        max_df=0.501,
    )
    # Construct POS TF matrix and get vocab dict
    pos = pos_vectorizer.fit_transform(pd.Series(tweet_tags)).toarray()
    pos_vocab = {v: i for i, v in enumerate(pos_vectorizer.get_feature_names())}
    feats = get_oth_features(tweets)
    # Now join them all up
    M = np.concatenate([tfidf, pos, feats], axis=1)
    print(M.shape)
    # Finally get a list of variable names
    variables = [''] * len(vocab)
    for k, v in vocab.items():
        variables[v] = k
    pos_variables = [''] * len(pos_vocab)
    for k, v in pos_vocab.items():
        pos_variables[v] = k
    all_feature_names = variables + pos_variables + other_features_names
    X = pd.DataFrame(M)
    return X, all_feature_names, vocab, idf_vals


def prepare_data_from_pkls(tweets, tf_vec, idf_vec, pos_vec, other_idxs):
    print(len(tweets), " tweets to classify")
    X = transform_inputs(tweets, tf_vec, idf_vec, pos_vec, other_idxs)
    return X


def train_model(X, y, feature_names, vocab, idf_vals, pkl_dir):
    select = SelectFromModel(LogisticRegression(class_weight='balanced', penalty="l1", C=0.01))
    X_ = select.fit_transform(X, y)
    model = LinearSVC(class_weight='balanced', C=0.01, penalty='l2', loss='squared_hinge', multi_class='ovr').fit(X_, y)
    y_preds = model.predict(X_)
    report = classification_report(y, y_preds)
    print(report)
    final_features = select.get_support(indices=True)  # get indices of features
    final_feature_list = [feature_names[i] for i in final_features]  # Get list of names corresponding to indices
    ngram_features, pos_features, oth_features = find_feature_split(final_feature_list)
    new_vocab = {v: i for i, v in enumerate(ngram_features)}
    new_vocab_to_index = {}
    for k in ngram_features:
        new_vocab_to_index[k] = vocab[k]
    ngram_indices = final_features[:len(ngram_features)]
    new_vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        preprocessor=preprocess,
        ngram_range=(1, 3),
        stop_words=stopwords,  # We do better when we keep stopwords
        use_idf=False,
        smooth_idf=False,
        norm=None,  # Applies l2 norm smoothing
        decode_error='replace',
        min_df=1,
        max_df=1.0,
        vocabulary=new_vocab
    )
    joblib.dump(new_vectorizer, os.path.join(pkl_dir, 'final_tfidf.pkl'))
    idf_vals_ = idf_vals[ngram_indices]
    joblib.dump(idf_vals_, os.path.join(pkl_dir, 'final_idf.pkl'))
    new_pos = {v: i for i, v in enumerate(pos_features)}
    new_pos_vectorizer = TfidfVectorizer(
        tokenizer=None,
        lowercase=False,
        preprocessor=None,
        ngram_range=(1, 3),
        stop_words=None,  # We do better when we keep stopwords
        use_idf=False,
        smooth_idf=False,
        norm=None,  # Applies l2 norm smoothing
        decode_error='replace',
        min_df=1,
        max_df=1.0,
        vocabulary=new_pos
    )
    joblib.dump(new_pos_vectorizer, os.path.join(pkl_dir, 'final_pos.pkl'))
    other_features_indices = [other_features_map[feature] for feature in oth_features]
    joblib.dump(other_features_indices, os.path.join(pkl_dir, 'other_idxs.pkl'))
    return model


def find_feature_split(feature_list):
    # VERY hacky way of finding the split of feature list
    pos_indices = [i for i in range(len(feature_list)) if feature_list[i][0].isupper()
                   and feature_list[i] not in ["FKRA", "FRE"]]
    ngram_features = feature_list[:pos_indices[0]]
    pos_features = feature_list[pos_indices[0]:pos_indices[-1]+1]
    oth_features = feature_list[pos_indices[-1]+1:]
    return ngram_features, pos_features, oth_features


def train(args):
    df = pd.read_csv(args.data_path, encoding='latin-1')
    X, feature_names, vocab, idf_vals = prepare_data_for_training(df[args.text_field])
    y = df['class'].astype(int)
    if not os.path.isdir(args.pkl_dir):
        os.mkdir(args.pkl_dir)
    model = train_model(X, y, feature_names, vocab, idf_vals, args.pkl_dir)
    with open(os.path.join(args.pkl_dir, args.model_name), 'wb') as f:
        pkl.dump(model, f)


def test(args):
    print("Calculate accuracy on labeled data")
    df = pd.read_csv(args.data_path, encoding='latin-1')
    tweets = df[args.text_field].values
    tweets = [x for x in tweets if type(x) == str]
    tweets_class = df['class'].values
    with open(os.path.join(args.pkl_dir, args.model_name), 'rb') as f:
        model = pkl.load(f)
    predictions = get_tweets_predictions(tweets, model, args.pkl_dir)
    if args.result_path:
        with open(args.result_path, 'w+') as f:
            f.write(classification_report(tweets_class, predictions))
    right_count = 0
    for i, t in enumerate(tweets):
        if tweets_class[i] == predictions[i]:
            right_count += 1

    accuracy = right_count / float(len(df))
    print("accuracy", accuracy)


def tag(args):
    with open(os.path.join(args.pkl_dir, args.model_name), 'rb') as f:
        model = pkl.load(f)

    print("Loading data to classify...")

    df = pd.read_csv(args.data_path, encoding='latin-1')
    input_texts = df[args.text_field]
    input_texts = [x for x in input_texts if type(x) == str]

    predictions = get_tweets_predictions(input_texts, model, args.pkl_dir)

    if args.result_path:
        with open(args.result_path, 'w+', encoding='latin-1') as f:
            for i, t in enumerate(input_texts):
                f.write("Text: {}\nClass: {}\n\n".format(t, class_to_name(predictions[i])))
    else:
        print("Printing predicted values: ")
        for i, t in enumerate(input_texts):
            print(t)
            print(class_to_name(predictions[i]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train model or classify tweets')
    parser.add_argument('--model_name',
                        default='model.pkl',
                        help="Name for model, to save (train) or load (test/tag)")
    parser.add_argument('--pkl_dir',
                        default='../models/',
                        help="Path to directory to save pickle of model and other files needed"
                             " to transform new data such that they can be classified.")
    parser.add_argument('--data_path',
                        default='trump_tweets.csv',
                        help='Path to data, must be CSV file')
    parser.add_argument('--text_field',
                        default='Text',
                        help="Name of field in data file in which to find the text data")
    parser.add_argument('--delimiter',
                        default=",",
                        help="Delimiter of data file")
    parser.add_argument('--result_path',
                        help="Path to save result; if not provided, results will be printed to stdout")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-r', '--train', action='store_true')
    group.add_argument('-e', '--test', action='store_true')
    group.add_argument('-a', '--tag', action='store_true')

    args, unknown = parser.parse_known_args()

    if args.train:
        train(args)

    elif args.tag:
        tag(args)

    elif args.test:
        test(args)
