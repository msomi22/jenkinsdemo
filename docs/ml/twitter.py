import tweepy
from tweepy import OAuthHandler
 
consumer_key = ''
consumer_secret = '' 
access_token = ''
access_secret = ''
 
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
 
api = tweepy.API(auth)


for status in tweepy.Cursor(api.home_timeline).items(10):
    # Process a single status
    print(status.text) 
    print('--------------------------------------------')


'''
Owner	geek_peter
'''
