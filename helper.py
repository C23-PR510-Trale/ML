import joblib
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore", "SettingWithCopyWarning")
pd.options.mode.chained_assignment = None 


def make_prediction(category, budget, num_of_recom):
    data = pd.DataFrame({"trip_id":{"0":1,"1":2,"2":3,"3":4,"4":5,"5":6,"6":7,"7":8,"8":9,"9":10,"10":11,"11":12,"12":13,"13":14,"14":15,"15":16,"16":17,"17":18,"18":19,"19":20,"20":21,"21":22,"22":23,"23":24,"24":25,"25":26,"26":27,"27":28,"28":29,"29":30,"30":31,"31":32,"32":33,"33":34,"34":35,"35":36,"36":37,"37":38,"38":39,"39":40,"40":41,"41":42,"42":43,"43":44,"44":45,"45":46,"46":47,"47":48,"48":49,"49":50,"50":51,"51":52,"52":53,"53":54,"54":55,"55":56,"56":57,"57":58,"58":59,"59":60,"60":61,"61":62,"62":63,"63":64,"64":65,"65":66,"66":67,"67":68,"68":69,"69":70,"70":71,"71":72,"72":73,"73":74,"74":75,"75":76,"76":77,"77":78,"78":79,"79":80,"80":81,"81":82,"82":83,"83":84,"84":85,"85":86,"86":87,"87":88,"88":89,"89":90,"90":91,"91":92,"92":93,"93":94,"94":95,"95":96,"96":97,"97":98,"98":99,"99":100},"trip_name":{"0":"Taste of Jakarta","1":"Historical Heritage","2":"Local Culture Experience","3":"Iconic Landmarks","4":"Culinary Delights","5":"Historical Journeys","6":"Cultural Immersion","7":"Landmark Adventures","8":"Gourmet Experiences","9":"Timeless Trails","10":"Cultural Exploration","11":"Architectural Wonders","12":"Taste of Malang","13":"Historical Wonders","14":"Cultural Festivals","15":"Iconic Structures","16":"Flavors of Jakarta","17":"Historical Gems","18":"Cultural Heritage","19":"Archaeological Sites","20":"Gourmet Adventure","21":"Timeless Tales","22":"Cultural Enrichment","23":"Landmark Discovery","24":"Taste of Yogyakarta","25":"Historical Exploration","26":"Cultural Discoveries","27":"Architectural Marvels","28":"Gourmet Delights","29":"Time Travelers","30":"Cultural Encounters","31":"Iconic Landscapes","32":"Flavors of Surabaya","33":"Historical Trails","34":"Local Traditions","35":"Architectural Marvels","36":"Taste of Bali","37":"Historical Heritage","38":"Cultural Journeys","39":"Landmark Expeditions","40":"Street Food Delights","41":"Timeless Treasures","42":"Cultural Encounters","43":"Iconic Landmarks","44":"Gourmet Escapades","45":"Historical Wonders","46":"Cultural Festivals","47":"Landmark Adventures","48":"Taste of Jakarta","49":"Historical Gems","50":"Local Culture Experience","51":"Architectural Marvels","52":"Gourmet Exploration","53":"Timeless Tales","54":"Cultural Heritage","55":"Iconic Structures","56":"Taste of Bandung","57":"Historical Journeys","58":"Cultural Exploration","59":"Archaeological Sites","60":"Gourmet Adventures","61":"Timeless Trails","62":"Cultural Enrichment","63":"Landmark Discovery","64":"Flavors of Bali","65":"Historical Exploration","66":"Cultural Discoveries","67":"Architectural Marvels","68":"Taste of Semarang","69":"Historical Gems","70":"Cultural Heritage","71":"Iconic Landmarks","72":"Street Food Delights","73":"Timeless Treasures","74":"Cultural Encounters","75":"Architectural Marvels","76":"Gourmet Escapades","77":"Historical Wonders","78":"Cultural Festivals","79":"Landmark Expeditions","80":"Taste of Surakarta","81":"Historical Trails","82":"Local Traditions","83":"Architectural Marvels","84":"Gourmet Adventures","85":"Timeless Tales","86":"Cultural Heritage","87":"Iconic Structures","88":"Taste of Jakarta","89":"Historical Journeys","90":"Cultural Exploration","91":"Archaeological Sites","92":"Gourmet Experiences","93":"Time Travelers","94":"Cultural Enrichment","95":"Landmark Discovery","96":"Flavors of Yogyakarta","97":"Historical Exploration","98":"Cultural Discoveries","99":"Architectural Marvels"},"category":{"0":0,"1":2,"2":1,"3":3,"4":0,"5":2,"6":1,"7":3,"8":0,"9":2,"10":1,"11":3,"12":0,"13":2,"14":1,"15":3,"16":0,"17":2,"18":1,"19":3,"20":0,"21":2,"22":1,"23":3,"24":0,"25":2,"26":1,"27":3,"28":0,"29":2,"30":1,"31":3,"32":0,"33":2,"34":1,"35":3,"36":0,"37":2,"38":1,"39":3,"40":0,"41":2,"42":1,"43":3,"44":0,"45":2,"46":1,"47":3,"48":0,"49":2,"50":1,"51":3,"52":0,"53":2,"54":1,"55":3,"56":0,"57":2,"58":1,"59":3,"60":0,"61":2,"62":1,"63":3,"64":0,"65":2,"66":1,"67":3,"68":0,"69":2,"70":1,"71":3,"72":0,"73":2,"74":1,"75":3,"76":0,"77":2,"78":1,"79":3,"80":0,"81":2,"82":1,"83":3,"84":0,"85":2,"86":1,"87":3,"88":0,"89":2,"90":1,"91":3,"92":0,"93":2,"94":1,"95":3,"96":0,"97":2,"98":1,"99":3},"budget":{"0":7.43432,"1":2.1534,"2":5.5274,"3":6.28543,"4":3.25931,"5":7.52191,"6":4.8295,"7":9.11237,"8":2.89789,"9":6.72903,"10":4.25689,"11":5.66742,"12":3.82548,"13":8.17642,"14":4.98352,"15":6.13902,"16":2.87513,"17":5.98281,"18":4.26842,"19":4.87694,"20":3.45219,"21":6.72359,"22":5.4329,"23":7.19381,"24":2.98491,"25":7.46972,"26":5.14839,"27":6.34519,"28":3.92431,"29":6.91902,"30":4.39271,"31":5.67912,"32":3.1589,"33":6.48127,"34":5.52713,"35":4.91083,"36":3.89125,"37":7.91432,"38":4.52379,"39":6.19241,"40":3.62415,"41":7.05943,"42":5.17693,"43":6.46298,"44":4.28694,"45":7.18231,"46":4.95317,"47":5.99813,"48":3.24827,"49":6.86194,"50":5.45369,"51":4.83982,"52":3.85267,"53":7.12934,"54":5.11322,"55":6.3394,"56":3.26816,"57":6.87532,"58":4.37621,"59":5.95862,"60":3.74593,"61":6.99381,"62":5.05189,"63":6.38241,"64":3.12409,"65":7.11273,"66":5.27413,"67":4.69641,"68":3.89752,"69":7.88314,"70":4.34593,"71":5.99419,"72":3.76182,"73":7.15982,"74":5.20976,"75":6.42327,"76":3.92418,"77":7.11882,"78":5.03724,"79":6.25181,"80":3.14072,"81":6.79412,"82":5.65912,"83":4.84216,"84":3.67594,"85":6.94831,"86":5.11239,"87":6.39571,"88":3.29518,"89":6.92134,"90":5.27491,"91":4.62137,"92":3.82579,"93":7.21094,"94":5.02491,"95":6.41259,"96":3.24179,"97":6.93825,"98":5.22351,"99":4.82361},"rating":{"0":3.5,"1":4.2,"2":4.6,"3":4.8,"4":3.1,"5":4.4,"6":4.9,"7":4.7,"8":3.2,"9":4.5,"10":4.8,"11":4.2,"12":3.7,"13":4.9,"14":4.1,"15":4.7,"16":3.8,"17":4.6,"18":4.5,"19":4.3,"20":2.9,"21":4.8,"22":4.4,"23":4.7,"24":3.3,"25":4.9,"26":4.1,"27":4.6,"28":3.7,"29":4.5,"30":4.3,"31":4.7,"32":3.2,"33":4.9,"34":4.1,"35":4.5,"36":3.5,"37":4.8,"38":4.2,"39":4.6,"40":3.4,"41":4.7,"42":4.3,"43":4.8,"44":3.6,"45":4.9,"46":4.1,"47":4.7,"48":3.2,"49":4.6,"50":4.5,"51":4.2,"52":3.8,"53":4.9,"54":4.3,"55":4.7,"56":3.5,"57":4.8,"58":4.1,"59":4.6,"60":3.4,"61":4.9,"62":4.3,"63":4.7,"64":3.2,"65":4.6,"66":4.5,"67":4.2,"68":3.8,"69":4.9,"70":4.1,"71":4.7,"72":3.4,"73":4.8,"74":4.3,"75":4.8,"76":3.6,"77":4.9,"78":4.1,"79":4.7,"80":3.3,"81":4.6,"82":4.5,"83":4.2,"84":3.8,"85":4.9,"86":4.3,"87":4.7,"88":3.2,"89":4.6,"90":4.5,"91":4.2,"92":3.8,"93":4.9,"94":4.1,"95":4.7,"96":3.3,"97":4.6,"98":4.5,"99":4.2},"location":{"0":"Jakarta","1":"Yogyakarta","2":"Surabaya","3":"Bandung","4":"Malang","5":"Bali","6":"Semarang","7":"Surakarta","8":"Jakarta","9":"Yogyakarta","10":"Surabaya","11":"Bandung","12":"Malang","13":"Bali","14":"Semarang","15":"Surakarta","16":"Jakarta","17":"Yogyakarta","18":"Surabaya","19":"Bandung","20":"Malang","21":"Bali","22":"Semarang","23":"Surakarta","24":"Yogyakarta","25":"Jakarta","26":"Surabaya","27":"Bandung","28":"Malang","29":"Bali","30":"Semarang","31":"Surakarta","32":"Surabaya","33":"Jakarta","34":"Yogyakarta","35":"Bandung","36":"Malang","37":"Bali","38":"Semarang","39":"Surakarta","40":"Jakarta","41":"Yogyakarta","42":"Surabaya","43":"Bandung","44":"Malang","45":"Bali","46":"Semarang","47":"Surakarta","48":"Jakarta","49":"Yogyakarta","50":"Surabaya","51":"Bandung","52":"Malang","53":"Bali","54":"Semarang","55":"Surakarta","56":"Bandung","57":"Jakarta","58":"Yogyakarta","59":"Surabaya","60":"Malang","61":"Bali","62":"Semarang","63":"Surakarta","64":"Bali","65":"Jakarta","66":"Yogyakarta","67":"Surabaya","68":"Semarang","69":"Bandung","70":"Malang","71":"Surakarta","72":"Jakarta","73":"Yogyakarta","74":"Surabaya","75":"Bandung","76":"Malang","77":"Bali","78":"Semarang","79":"Surakarta","80":"Surakarta","81":"Jakarta","82":"Yogyakarta","83":"Surabaya","84":"Bandung","85":"Malang","86":"Bali","87":"Semarang","88":"Jakarta","89":"Yogyakarta","90":"Surabaya","91":"Bandung","92":"Malang","93":"Bali","94":"Semarang","95":"Surakarta","96":"Yogyakarta","97":"Jakarta","98":"Surabaya","99":"Bandung"},"cluster":{"0":1,"1":2,"2":0,"3":1,"4":2,"5":1,"6":0,"7":1,"8":2,"9":1,"10":0,"11":1,"12":2,"13":1,"14":0,"15":1,"16":2,"17":1,"18":0,"19":0,"20":2,"21":1,"22":0,"23":1,"24":2,"25":1,"26":0,"27":1,"28":2,"29":1,"30":0,"31":1,"32":2,"33":1,"34":0,"35":0,"36":2,"37":1,"38":0,"39":1,"40":2,"41":1,"42":0,"43":1,"44":2,"45":1,"46":0,"47":1,"48":2,"49":1,"50":0,"51":0,"52":2,"53":1,"54":0,"55":1,"56":2,"57":1,"58":0,"59":1,"60":2,"61":1,"62":0,"63":1,"64":2,"65":1,"66":0,"67":0,"68":2,"69":1,"70":0,"71":1,"72":2,"73":1,"74":0,"75":1,"76":2,"77":1,"78":0,"79":1,"80":2,"81":1,"82":0,"83":0,"84":2,"85":1,"86":0,"87":1,"88":2,"89":1,"90":0,"91":0,"92":2,"93":1,"94":0,"95":1,"96":2,"97":1,"98":0,"99":0}})
    
    label_encoder = joblib.load("label_encoder.h5")
    kmeans = joblib.load("kmeans.h5")

    new_data = pd.DataFrame({
        'category': [category],
        'budget': [budget]
    })
    new_data['budget'] = new_data['budget'] / 100000

    # Perform label encoding for the new data
    new_data['category'] = label_encoder.transform(new_data['category'])

    # Predict the cluster label for the new data
    new_data['cluster'] = kmeans.predict(new_data[['budget', 'category']])
    # Retrieve the cluster label for the new data
    new_data_cluster = new_data['cluster'].iloc[0]

    filtered_data = data[data['cluster'] == new_data_cluster]

    # Sort the filtered data based on the Euclidean distance from the new data point
    filtered_data['distance'] = ((filtered_data['budget'] - new_data['budget'].iloc[0])**2 +
                                 (filtered_data['category'] - new_data['category'].iloc[0])**2)**0.5
    filtered_data = filtered_data.sort_values('distance')

    # Top K predictions based on the sorted distances
    K = num_of_recom
    top_K_points = filtered_data.head(K)
    
    trip_id = top_K_points["trip_id"].to_numpy()
    trip_name = top_K_points["trip_name"].to_numpy()
    budget = top_K_points["budget"].to_numpy()
    rating = top_K_points["rating"].to_numpy()
    location = top_K_points["location"].to_numpy()

    trip_id = trip_id.tolist()
    trip_name = trip_name.tolist()
    budget = budget.tolist()
    rating = rating.tolist()
    location = location.tolist()

    output = {
        "trip_id":trip_id,
        "trip_name":trip_name,
        "budget":budget,
        "rating":rating,
        "location":location
    }

    return output
