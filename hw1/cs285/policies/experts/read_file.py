import pickle

def open_test_data():
    return open('./Ant.pkl', 'rb')

with open_test_data() as f:
    dict1 = pickle.load(f)

print(dict1)
