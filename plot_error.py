import matplotlib.pyplot as plt
import pickle

filename = '/Users/adityajain/Downloads/error_data/city_sift_10000.pkl'
data     = pickle.load(open(filename, 'rb'))

print(data)
