import numpy as np
import pandas as pd

__author__ = "Devanshi Prajapati"

def web_scrapping(url, classname):
    import requests
    from bs4 import BeautifulSoup

    r = requests.get(url)
    
    if r.status_code == 200:
        soup = BeautifulSoup(r.text, 'html.parser')
        
        reviews = soup.findAll(class_=classname)
        
        review_list = []
        for review in reviews:
            review_text = review.get_text()
            review_list.append(review_text)
        return review_list
    
    return []  


def preprocessing(reviews):
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    reviews['cleaned_review'] = reviews['review']

    reviews['cleaned_review'] = reviews['cleaned_review'].str.lower()

    reviews['cleaned_review'] = reviews['cleaned_review'].str.replace(r'[^\w\s]', '', regex=True)

    nltk.download('stopwords', quiet=True)

    stop_words = set(stopwords.words('english'))
    for word in stop_words:
        reviews['cleaned_review'] = reviews['cleaned_review'].apply(lambda x: ' '.join([w for w in x.split() if w != word]))

    def lemmatizer_func(x):
        lemmatizer = WordNetLemmatizer()
        words = x.split()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        return ' '.join(lemmatized_words)

    reviews['cleaned_review'] = reviews['cleaned_review'].apply(lemmatizer_func)

    return reviews 


if __name__ == '__main__':
    # give your desired urls and classnames, preferably from yelp
    url1, url2 = "https://www.yelp.com/biz/vans-tempe-3", "https://www.yelp.com/biz/urban-outfitters-phoenix-2?osq=Shopping"
    classname1, classname2 = "comment__09f24__D0cxf", "comment__09f24__D0cxf"

    # Part 1
    review_list1 = web_scrapping(url1, classname1)
    review_list2 = web_scrapping(url2, classname2)

    # Create a pandas dataframe from array
    df1 = pd.DataFrame(np.array(review_list1), columns=['review'])
    df2 = pd.DataFrame(np.array(review_list2), columns=['review'])

    # Part 2
    processed_review1 = preprocessing(df1)
    processed_review2 = preprocessing(df2)
    #print(df1) #To print the dataframes as output by the function
    #print(df2)

    # Code to display the reviews on console as shown in sample output
    def display_processed_reviews(reviews, store_name):
        print(f"\nProcessed reviews for {store_name}:")
        #Printing each review within 1 line 
        for i, review in enumerate(reviews['cleaned_review'], 1):
            if len(review) > 80:
                short_review = review[:80-3] + '...'
                print(f"{i} {short_review}")
            else:
                print(f"{i} {review}")
            
    display_processed_reviews(processed_review1, "Vans")
    display_processed_reviews(processed_review2, "Urban Outfitters")
