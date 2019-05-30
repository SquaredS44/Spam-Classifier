import re
from nltk.stem import PorterStemmer


def has_numbers(input_string):
    return any(char.isdigit() for char in input_string)


def get_vocab():  # vocabulary list (from txt file)
    vocab_list = []
    with open('vocab.txt', 'r') as f:
        for line in f:
            for word in line.split():
                if not has_numbers(word):
                    vocab_list.append(word)
    f.close()
    return vocab_list


def clean_html(raw_html):
    to_remove = re.compile('<.*?>')
    clean_text = re.sub(to_remove, '', raw_html)
    return clean_text


def clean_numbers(raw_numbers):
    clean_text = re.sub('\d+', 'number', raw_numbers)
    return clean_text


def clean_urls(raw_urls):
    to_remove = re.compile(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b')
    clean_text = re.sub(to_remove, 'httpaddr', raw_urls)
    return clean_text


def clean_emails(raw_emails):
    to_remove = re.compile(r'[^ ]+@[^ ]+')
    clean_text = re.sub(to_remove, 'emailaddr', raw_emails)
    return clean_text


def clean_currency(raw_currency):
    to_remove_dollar = re.compile(r'[$]+')
    clean_text = re.sub(to_remove_dollar, 'currency', raw_currency)
    to_remove_euro = re.compile(r'[€]+')
    clean_text = re.sub(to_remove_euro, 'currency', clean_text)
    to_remove_pound = re.compile(r'[£]+')
    clean_text = re.sub(to_remove_pound, 'currency', clean_text)
    return clean_text


def processEmail(email_cont):  # email cont = email sample
    # remove header
    # lower case
    email_cont = email_cont.lower()
    # strip all HTML
    email_cont = clean_html(email_cont)
    # look for numbers
    email_cont = clean_numbers(email_cont)
    # handle URLs
    email_cont = clean_urls(email_cont)
    # handle email addresses
    email_cont = clean_emails(email_cont)
    # handle dollar, euro, pound sign
    email_cont = clean_currency(email_cont)
    email_token = token_stem(email_cont)
    index_list = index_email(email_token)
    return index_list


def token_stem(processed_email):  # processed_email = processEmail(email_cont)
    # tokenize
    # split the email into individual words (tokens) (split by the delimiter ' ')
    # splitting by many delimiters is easiest with re.split() and removes any punctuation
    tokens = re.split('[ \@\$\/\#\.\-\:\&\*\+\=\[\]\?\!\(\)\{\}\,\'\"\>\_\<\;\%]', processed_email)
    # stem
    stemmer = PorterStemmer()
    token_list = []
    for token in tokens:
        #  remove any non-alphanumeric character
        token = re.sub('[^a-zA-Z0-9]', '', token)
        if not len(token):
            continue
        token_list.append(stemmer.stem(token))
    return token_list


def index_email(token_list):  # token_list = token_stem(processed_email)
    vocab_list = get_vocab()
    index_list = []
    for word in token_list:
        if word in vocab_list:
            index_list.append(vocab_list.index(word))
    return index_list
