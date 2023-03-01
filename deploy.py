from flask import Flask, render_template, request
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
import re
import pandas as pd


app = Flask(__name__)

y_label_question_names = ['question_asker_intent_understanding', 'question_body_critical', 'question_conversational', 'question_expect_short_answer', 'question_fact_seeking', 'question_has_commonly_accepted_answer', 'question_interestingness_others', 'question_interestingness_self', 'question_multi_intent', 'question_not_really_a_question', 'question_opinion_seeking', 'question_type_choice', 'question_type_compare', 'question_type_consequence', 'question_type_definition', 'question_type_entity', 'question_type_instructions', 'question_type_procedure', 'question_type_reason_explanation', 'question_type_spelling', 'question_well_written']
y_label_answer_names = ['answer_helpful', 'answer_level_of_information', 'answer_plausible', 'answer_relevance', 'answer_satisfaction', 'answer_type_instructions', 'answer_type_procedure', 'answer_type_reason_explanation', 'answer_well_written']



def decontracted(phrase):
    phrase = re.sub(r"(W|w)on(\'|\’)t ", "will not ", phrase)
    phrase = re.sub(r"(C|c)an(\'|\’)t ", "can not ", phrase)
    phrase = re.sub(r"(Y|y)(\'|\’)all ", "you all ", phrase)
    phrase = re.sub(r"(Y|y)a(\'|\’)ll ", "you all ", phrase)
    phrase = re.sub(r"(I|i)(\'|\’)m ", "i am ", phrase)
    phrase = re.sub(r"(A|a)isn(\'|\’)t ", "is not ", phrase)
    phrase = re.sub(r"n(\'|\’)t ", " not ", phrase)
    phrase = re.sub(r"(\'|\’)re ", " are ", phrase)
    phrase = re.sub(r"(\'|\’)d ", " would ", phrase)
    phrase = re.sub(r"(\'|\’)ll ", " will ", phrase)
    phrase = re.sub(r"(\'|\’)t ", " not ", phrase)
    phrase = re.sub(r"(\'|\’)ve ", " have ", phrase)
    return phrase

def clean_text(x):
    x = str(x)
    for punct in "/-'":
        x = x.replace(punct, ' ')
    for punct in '&':
        x = x.replace(punct, f' {punct} ')
    for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':
        x = x.replace(punct, '')
    return x

def clean_numbers(x):
    x = re.sub('[0-9]{5,}', '12345', x)
    x = re.sub('[0-9]{4}', '1234', x)
    x = re.sub('[0-9]{3}', '123', x)
    x = re.sub('[0-9]{2}', '12', x)
    return x

# https://gist.github.com/sebleier/554280
# we are removing the words from the stop words list: 'no', 'nor', 'not'
stopwords= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"]

stemmer = SnowballStemmer("english")

# Combining all the above stundents 
from tqdm import tqdm
def preprocess_text(text_data):
    preprocessed_text = []
    # tqdm is for printing the status bar
    for sentance in tqdm(text_data):
        sent = decontracted(sentance)
        sent = clean_text(sentance)
        sent = clean_numbers(sentance)
        sent = sent.replace('\\r', ' ')
        sent = sent.replace('\\n', ' ')
        sent = sent.replace('\\"', ' ')
        sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
        # https://gist.github.com/sebleier/554280
        sent = ' '.join(stemmer.stem(e) for e in sent.split() if e.lower() not in stopwords)
        preprocessed_text.append(sent.lower().strip())
    return preprocessed_text


def special_symbols(string):
    pattern = r'[^a-zA-Z0-9\s]'
    matches = re.finditer(pattern, string, re.MULTILINE)
    return len([_ for _ in matches])


dbfile = open('encoders/host_encoder.pkl', 'rb')     
host_encoder = pickle.load(dbfile)
dbfile.close()

dbfile = open('encoders/category_encoder.pkl', 'rb')     
category_encoder = pickle.load(dbfile)
dbfile.close()


dbfile = open('encoders/preprocessed_question.pkl', 'rb')     
preprocessed_question_vectorizer = pickle.load(dbfile)
dbfile.close()

dbfile = open('encoders/preprocessed_answer.pkl', 'rb')     
preprocessed_answer_vectorizer = pickle.load(dbfile)
dbfile.close()



models_dict = {}
for aspect in y_label_question_names+y_label_answer_names:

    print("Model for aspect :", aspect)
    filename = f'RF_{aspect}'
    print('Model Exists!')
    sig_clf = pickle.load(open('models/' + filename, 'rb'))
    models_dict[aspect] = sig_clf



@app.route('/')
def home():
    result = ''
    return render_template('index.html', **locals())

@app.route('/predict', methods=['POST', "GET"])
def predict():
    
    question_title = str(request.form['question_title'])
    question_body = str(request.form['question_body'])
    question_user_name = str(request.form['question_user_name'])
    question_user_page = str(request.form['question_user_page'])
    answer = str(request.form['answer'])
    answer_user_name = str(request.form['answer_user_name'])
    answer_user_page = str(request.form['answer_user_page'])
    url = str(request.form['url'])
    category = str(request.form['category'])
    host = str(request.form['host'])

    df = pd.DataFrame({'question_title':[question_title], 
                        'question_body':[question_body],
                        'answer':[answer],
                        'category':[category],
                        'host':[host]
                        })
    
    #######################################################################################################
    df['preprocessed_question_title'] = preprocess_text(df['question_title'].values)
    df['preprocessed_question_body'] = preprocess_text(df['question_body'].values)
    df['preprocessed_answer'] = preprocess_text(df['answer'].values)
    #######################################################################################################
    df['question_title_length'] = df['preprocessed_question_title'].apply(lambda x: len(x.split()))
    #######################################################################################################
    df['question_body_length'] = df['preprocessed_question_body'].apply(lambda x: len(x.split()))
    #######################################################################################################
    df['answer_length'] = df['preprocessed_answer'].apply(lambda x: len(x.split()))
    #######################################################################################################
    df['question_title_special_symbols'] = df['question_title'].apply(special_symbols)
    df['question_body_special_symbols'] = df['question_body'].apply(special_symbols)
    df['answer_special_symbols'] = df['answer'].apply(special_symbols)
    #######################################################################################################

    df['preprocessed_question'] = df['preprocessed_question_title'] + df['preprocessed_question_body']

    df_host = host_encoder.transform(df['host'])
    df_category = category_encoder.transform(df['category'])

    df_question_tfidf = preprocessed_question_vectorizer.transform(df['preprocessed_question'])
    df_answer_tfidf = preprocessed_answer_vectorizer.transform(df['preprocessed_answer'])


    print(df_question_tfidf.toarray().shape, df_host.reset_index(drop = True).shape, df_category.reset_index(drop = True).shape, df[['question_title_length', 'question_body_length', 'answer_length', 'question_title_special_symbols', 'question_body_special_symbols',	'answer_special_symbols']].reset_index(drop = True).shape)

    # question based feature
    df_A = pd.concat([pd.DataFrame(df_question_tfidf.toarray()), df_host.reset_index(drop = True), df_category.reset_index(drop = True), df[['question_title_length', 'question_body_length', 'answer_length', 'question_title_special_symbols', 'question_body_special_symbols',	'answer_special_symbols']].reset_index(drop = True)], axis=1)
    # answer based feature
    df_B = pd.concat([pd.DataFrame(df_answer_tfidf.toarray()), df_host.reset_index(drop = True), df_category.reset_index(drop = True), df[['question_title_length', 'question_body_length', 'answer_length', 'question_title_special_symbols', 'question_body_special_symbols',	'answer_special_symbols']].reset_index(drop = True)], axis=1)

    print(df_A.shape)
    print(df_B.shape)

    pred = []
    for aspect in y_label_question_names+y_label_answer_names:
        if 'question' in aspect:
            pred.append(models_dict[aspect].predict(df_A))
        else:
            pred.append(models_dict[aspect].predict(df_B))
    print(pred)
    result  =  '<br>'
    result +=  'question_asker_intent_understanding : '+ str(1/9*pred[0][0]) +'<br>'
    result +=  'question_body_critical : '+ str(1/9*pred[1][0]) +'<br>'
    result +=  'question_conversational : '+ str(1/5*pred[2][0]) +'<br>'
    result +=  'question_expect_short_answer : '+ str(1/5*pred[3][0]) +'<br>'
    result +=  'question_fact_seeking : '+ str(1/5*pred[4][0]) +'<br>'
    result +=  'question_has_commonly_accepted_answer : '+ str(1/5*pred[5][0]) +'<br>'
    result +=  'question_interestingness_others : '+ str(1/9*pred[6][0]) +'<br>'
    result +=  'question_interestingness_self : '+ str(1/9*pred[7][0]) +'<br>'
    result +=  'question_multi_intent : '+ str(1/5*pred[8][0]) +'<br>' 
    result +=  'question_not_really_a_question : '+ str(1/5*pred[9][0]) +'<br>'
    result +=  'question_opinion_seeking : '+ str(1/5*pred[10][0]) +'<br>'
    result +=  'question_type_choice : '+ str(1/5*pred[11][0]) +'<br>'
    result +=  'question_type_compare : '+ str(1/5*pred[12][0]) +'<br>'
    result +=  'question_type_consequence : '+ str(1/5*pred[13][0]) +'<br>'
    result +=  'question_type_definition : '+ str(1/5*pred[14][0]) +'<br>'
    result +=  'question_type_entity : '+ str(1/5*pred[15][0]) +'<br>'
    result +=  'question_type_instructions : '+ str(1/5*pred[16][0]) +'<br>'
    result +=  'question_type_procedure : '+ str(1/5*pred[17][0]) +'<br>'    
    result +=  'question_type_reason_explanation : '+ str(1/5*pred[18][0]) +'<br>'
    result +=  'question_type_spelling : '+ str(1/3*pred[19][0]) +'<br>'
    result +=  'question_well_written : '+ str(1/9*pred[20][0]) +'<br>'

    result +=  'answer_helpful : '+ str(1/9*pred[21][0]) +'<br>'
    result +=  'answer_level_of_information : '+ str(1/9*pred[22][0]) +'<br>'
    result +=  'answer_plausible : '+ str(1/9*pred[23][0]) +'<br>'
    result +=  'answer_relevance : '+ str(1/9*pred[24][0]) +'<br>'    
    result +=  'answer_satisfaction : '+ str(1/17*pred[25][0]) +'<br>'
    result +=  'answer_type_instructions : '+ str(1/5*pred[26][0]) +'<br>'
    result +=  'answer_type_procedure : '+ str(1/5*pred[27][0]) +'<br>'
    result +=  'answer_type_reason_explanation : '+ str(1/5*pred[28][0]) +'<br>'
    result +=  'answer_well_written : '+ str(1/9*pred[29][0]) +'<br>'


    return render_template('index.html', **locals())


if __name__ == '__main__':
    app.run(debug=True)