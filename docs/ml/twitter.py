import tweepy
from tweepy import OAuthHandler
 
consumer_key = 'ciUdjEEdR5RHm0bZanrygrK8W'
consumer_secret = 'QadIHz2Xa9kba790sqP4IsSwtS8ARyMA917K4oZaUFL29GNnrM' 
access_token = '410586335-4XBelCaBXJ5MSu6S960tiwczOvtbGnJ57wMsHkt1'
access_secret = 'c3TADrXiCKZiTR7tiJ5vpuDBUxZP2XzF6i0iJycAaNLxg'
 
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
 
api = tweepy.API(auth)


for status in tweepy.Cursor(api.home_timeline).items(10):
    # Process a single status
    print(status.text) 
    print('--------------------------------------------')


'''
Consumer Key (API Key)	ciUdjEEdR5RHm0bZanrygrK8W
Consumer Secret (API Secret)	QadIHz2Xa9kba790sqP4IsSwtS8ARyMA917K4oZaUFL29GNnrM

Access Token	410586335-4XBelCaBXJ5MSu6S960tiwczOvtbGnJ57wMsHkt1
Access Token Secret	c3TADrXiCKZiTR7tiJ5vpuDBUxZP2XzF6i0iJycAaNLxg
Access Level	Read and write
Owner	geek_peter
Owner ID	410586335
'''
