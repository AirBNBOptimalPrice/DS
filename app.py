from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

# Initialize Flask app
app = Flask(__name__)

# load pickled model and encoder
model = pickle.load(open('finalized_model.pkl','rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))

# Function used for processing description
STOPWORDS = set(STOPWORDS).union(set(['and']))
def tokenize(text):
    return [token for token in simple_preprocess(text) if token not in STOPWORDS]


import requests
@app.route('/', methods=['POST', 'GET'])
def responses():
    response = requests.get('https://bnb-web-backend.herokuapp.com/api/features')
    return str(response.text)


@app.route('/api', methods=['POST', 'GET'])
def predict():
    """
    Receives JSON as input and outputs optimal price
    """
    
    # retrieve json user input data
    data = request.get_json(force=True)

    # Assign incoming data to variables for use in model
    desc = data['feature'][0]['description'] 

    # Run description through tokenize function
    description_length = len(tokenize(desc))

    # Continue assigning incoming data to variables for use in model
    neighbourhood_group_cleansed = data['feature'][0]['neighbourhood_group_cleansed']
    property_type = data['feature'][0]['property_type']
    accommodates = data['feature'][0]['accommodates']
    bathrooms = data['feature'][0]['bathrooms'] 
    bedrooms = data['feature'][0]['bedrooms'] 
    security_deposit = data['feature'][0]['security_deposit'] 
    cleaning_fee = data['feature'][0]['cleaning_fee']                   
    guests_included = data['feature'][0]['guests_included']                
    extra_people = data['feature'][0]['extra_people']                   
    minimum_nights = data['feature'][0]['minimum_nights']                   
    instant_bookable = data['feature'][0]['instant_bookable']                 
    cancellation_policy = data['feature'][0]['cancellation_policy']          
    tv_cable = data['feature'][0]['tv_cable']                             
    pets_allowed = data['feature'][0]['pets_allowed']                      

    # convert to list for model
    
    feature_dict = {'neighbourhood_group_cleansed':neighbourhood_group_cleansed, 
                    'property_type':property_type, 
                    'accommodates':accommodates,
                    'bathrooms':bathrooms, 
                    'bedrooms':bedrooms, 
                    'security_deposit':security_deposit, 
                    'cleaning_fee':cleaning_fee,
                    'guests_included':guests_included, 
                    'extra_people':extra_people, 
                    'minimum_nights':minimum_nights, 
                    'instant_bookable':instant_bookable,
                    'cancellation_policy':cancellation_policy, 
                    'description_length':description_length, 
                    'tv':tv_cable, 
                    'pets':pets_allowed}

    features = pd.DataFrame(feature_dict, index=[1])
    
    encoded_features = np.array([neighbourhood_group_cleansed, property_type,
                        instant_bookable, cancellation_policy,
                        tv_cable, pets_allowed, bathrooms]).reshape(1, -1)
    print(features)
    # run the features through the encoder
    features_transformed = encoder.transform(features)
   # print(features_transformed)
    
    # predict optimal price using the prediction function
    price = model.predict(features_transformed)
    print(price.shape)
    # return the optimal price in json format
    return jsonify({'prediction': price[0]})




if __name__ == "__main__":

    # TODO
    app.run(debug = True)
