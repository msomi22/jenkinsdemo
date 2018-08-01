#import nltk 
#nltk.download('punkt') 

from nltk import word_tokenize

text = ['peter is happy','this is a happy day','it was such a boring class']

for item in text:
   tokens = word_tokenize(item)
   print(item)






