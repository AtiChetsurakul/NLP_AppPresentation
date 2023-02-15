import praw
from secKey import ret_id
import pycache_loader_


if __name__ == '__main__':

    client, secret, _ = ret_id()
    reddit = praw.Reddit(client_id=client,

                         client_secret=secret,

                         user_agent='ati')

    print(reddit.read_only)  # Output: True

    subreddit = reddit.subreddit('apple')
    topics = [*subreddit.top(limit=10)]  # top posts all time
    # print(len(topics))
    fifty_sen = [n.title for n in topics]
    print(test := pycache_loader_.inference_classification(fifty_sen, False))

    print(list([int(i) for i in test]))
