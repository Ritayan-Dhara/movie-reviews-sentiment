from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import re
import requests
from bs4 import BeautifulSoup

from textblob import TextBlob

import pickle


with warnings.catch_warnings():
      warnings.simplefilter("ignore", category=UserWarning)
      clf = pickle.load(open('sentiment_model .pkl', 'rb'))
      cv = pickle.load(open('CountVectorizer.pkl', 'rb'))
#clf.__getstate__()['_sklearn_version']
#cv.__getstate__()['_sklearn_version']


data = pd.read_csv('movie_list.csv')


"""import sklearn
print('sklearn: %s' % sklearn.__version__)"""


app = Flask(__name__)


#clf = pickle.load(open('sentiment_model .pkl','rb'))
#cv = pickle.load(open('CountVectorizer.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    title = request.form.get("title")
    id = data[(data['title'] == title) | (data['original_title'] == title)]['imdb_title_id'].values[0]

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.162 Safari/537.36'}
    webpage = requests.get('https://www.imdb.com/title/{}/reviews'.format(id), headers=headers).text
    soup = BeautifulSoup(webpage, 'lxml')
    reviews = soup.find_all('div', attrs={'class': ['lister-item mode-detail', 'imdb-user-review']})

    review = []
    title = []
    rating = []
    by = []
    date = []
    for i in reviews:
        by.append(i.find_all('span', class_='display-name-link')[0].text.strip())
        date.append(i.find_all('span', class_='review-date')[0].text.strip())
        title.append(i.find_all('a', class_='title')[0].text.strip())
        review.append(i.find_all('div', class_='text show-more__control')[0].text.strip())
        rating.append(i.find_all('div', class_='review-container')[0].find('span', attrs={'class': None}).text.strip())

    d = {'IMDb id': id, 'rating': rating, 'title': title, 'review': review, 'by': by, 'date': date}
    reviews_list = pd.DataFrame(d)

    reviews_list['sentiment'] = reviews_list['review'].apply(pred)

    print(reviews_list)



    return render_template('reviews.html', label=reviews_list)


def pred(rev):
    rev = TextBlob(rev)
    polarity = rev.sentiment.polarity

    if(polarity <= 0.00):
        return 0
    else:
        return 1



if __name__=="__main__":
    app.run(debug=True)