import pickle
import sys
import os

file_name = sys.argv[1]

print("Load file: %s" %file_name)

file = pickle.load(open(file_name, 'rb'))

print(file)

