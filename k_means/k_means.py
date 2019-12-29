import math
import sys
import re
import io
import random as rd


regex_str = [
    r'<[^>]+>',
    r'(?:@[\w_]+)',
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',
    r'(?:(?:\d+,?)+(?:\.?\d+)?)',
    r"(?:[a-z][a-z'\-_]+[a-z])",
    r'(?:[\w_]+)',
]

tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)


def tokenize(s):
    return tokens_re.findall(s)


def preprocessing(filename, lowercase=True):
    f = open(filename, 'r', encoding='ISO-8859-1')
    tweets = list(f)
    list_of_processed_tweets = list()
    tweets_len = len(tweets)
    for i in range(tweets_len):
        # remove \n from the end after every sentence
        tweets[i] = tweets[i].strip('\n')

        # Remove the tweet id and timestamp
        tweets[i] = tweets[i][50:]

        # Remove any word that starts with the symbol @
        tweets[i] = " ".join(filter(lambda x: x[0] != '@', tweets[i].split()))

        # Remove any hash-tags symbols
        tweets[i] = tweets[i].replace('#', '')

        # Remove any URL
        tweets[i] = re.sub(r"http\S+", "", tweets[i])
        tweets[i] = re.sub(r"www\S+", "", tweets[i])

        # every word in tweet to lowercase
        tweets[i] = tweets[i].lower()

        # trim extra spaces
        tweets[i] = " ".join(tweets[i].split())

        # append each tweet to a list of processed tweets
        list_of_processed_tweets.append(tweets[i].split(' '))

    f.close()
    return list_of_processed_tweets


# defining the Jaccard distance
def get_jaccard_distance(tweet1, tweet2):
    intersection = list(set(tweet1) & set(tweet2))
    I_len = len(intersection)
    union = list(set(tweet1) | set(tweet2))
    U_len = len(union)
    return (1-(float(I_len)/U_len))


def assign_cluster(tweets, centroids):

    clusters = dict()

    # for every tweet, iterate each tweet and assign it to closest centroid
    for t in range(len(tweets)):
        min_distance = math.inf
        cluster_index = -1;
        for c in range(len(centroids)):
            distance = get_jaccard_distance(centroids[c], tweets[t])
            # look for a closest centroid for a tweet
            if distance < min_distance:
                cluster_index = c
                min_distance = distance

        # randomly assign the centroid to a tweet if nothing is common
        if min_distance == 1:
            cluster_index = rd.randint(0, len(centroids) - 1)

        # assign the closest centroid to a tweet
        clusters.setdefault(cluster_index, []).append([tweets[t]])

        # add the tweet distance from its closest centroid to compute sse
        last_tweet_idx = len(clusters.setdefault(cluster_index, [])) - 1
        clusters.setdefault(cluster_index, [])[last_tweet_idx].append(min_distance)

    return clusters


def compute_SSE(clusters):
    sse = 0

    # compute SSE as the sum of square of distances of the tweet from it's centroid through iterating every cluster
    for c in range(len(clusters)):
        for t in range(len(clusters[c])):
            sse = sse + (clusters[c][t][1] * clusters[c][t][1])

    return sse


def update_centroids(clusters):

    centroids = []

    for c in range(len(clusters)):
        min_distance_sum = math.inf
        centroid_idx = -1

        # list to store pre-calculated distances to avoid redundant calculations
        min_distance_list = []

        for l in range(len(clusters[c])):
            min_distance_list.append([])
            distance_sum = 0
            # get distances sum for every tweet t1 with every tweet t2 in a same cluster
            for m in range(len(clusters[c])):
                if l != m:
                    if m < l:
                        distance = min_distance_list[m][l]
                    else:
                        distance = get_jaccard_distance(clusters[c][l][0], clusters[c][m][0])

                    min_distance_list[l].append(distance)
                    distance_sum += distance
                else:
                    min_distance_list[l].append(0)

            # select the tweet with the minimum distances sum as the centroid for the cluster
            if distance_sum < min_distance_sum:
                min_distance_sum = distance_sum
                centroid_index = l

        # append the selected tweet to the centroid list
        centroids.append(clusters[c][centroid_index][0])

    return centroids


def is_converged(previous_centroid, new_centroids):

    # false if lengths are not equal
    if len(previous_centroid) != len(new_centroids):
        return False

    # iterate over each entry of clusters and check if they are same
    for c in range(len(new_centroids)):
        if " ".join(new_centroids[c]) != " ".join(previous_centroid[c]):
            return False

    return True


def train(tweets, k, max_iterations=5):
    centroids = []

    # initialization, assign random tweets as centroids
    for i in range(k):
        random_tweet_index = rd.randint(0, len(tweets) - 1)
        centroids.append(tweets[random_tweet_index])

    previous_centroids = []
    iteration_count = 0

    # Run iterations until converged or reached to max_iterations
    while(is_converged(previous_centroids, centroids)) == False and (iteration_count < max_iterations):
        print("running iteration ", iteration_count)

        clusters = assign_cluster(tweets, centroids)

        # if k_means converges, keep track of prev_centroids
        previous_centroids = centroids

        # update, update centroid based on clusters formed
        centroids = update_centroids(clusters)
        iteration_count = iteration_count + 1

    if iteration_count == max_iterations:
        print("max iterations reached, K-means not converged")
    else:
        print("K-means converged")

    sse = compute_SSE(clusters)

    return clusters, sse



