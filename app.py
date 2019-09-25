from flask import Flask, request, jsonify, render_template
import numpy as np
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


@app.route('/', methods=['POST', 'GET'])
def predict():
    """
    Receives JSON as input and outputs optimal price
    """

    # retrieve json user input data
    data = request.get_json(force=True)

    # Assign incoming data to variables for use in model
    desc = req_data['description'] 

    # Run description through tokenize function
    description = tokenize(desc)

    # Continue assigning incoming data to variables for use in model
    neighbourhood_group_cleansed = req_data['neighbourhood_group_cleansed']
    property_type = req_data['property_type']
    accommodates = req_data['accommodates']
    bathrooms = req_data['bathrooms'] 
    bedrooms = req_data['bedrooms'] 
    security_deposit = req_data['security_deposit'] 
    cleaning_fee = req_data['cleaning_fee']                   
    guests_included = req_data['guests_included']                
    extra_people = req_data['extra_people']                   
    minimum_nights = req_data['minimum_nights']                   
    instant_bookable = req_data['instant_bookable']                 
    cancellation_policy = req_data['cancellation_policy']          
    tv_cable = req_data['tv_cable']                             
    pets_allowed = req_data['pets_allowed']                      

    # convert to list for model
    features = [description, neighbourhood_group_cleansed, property_type, accommodates,
                bathrooms, bedrooms, security_deposit, cleaning_fee, guests_included,
                extra_people, minimum_nights, instant_bookable, cancellation_policy, 
                tv_cable, pets_allowed]
    
    # run the features through the encoder
    features_transformed = encoder.transform(features)

    # predict optimal price using the prediction function
    price = model.predict(features_transformed)

# return the optimal price in json format
return  jsonify({'prediction': price})




if __name__ == "__main__":

    # TODO
    app.run(debug = True)
