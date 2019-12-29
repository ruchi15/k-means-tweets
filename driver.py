from k_means.k_means import preprocessing
from k_means.k_means import train


if __name__ == '__main__':
    filename = 'data/bbchealth.txt'

    processed_tweets = preprocessing(filename)

    # number of experiments to be performed
    no_of_experiments = 5

    # starting value of K for K-means, it will keep increasing for the number of experiments
    k = 3

    # for every experiment, run K-means
    for e in range(no_of_experiments):

        print("Running K-means for experiment no. " + str(e+1) + " for k = " + str(k))

        clusters, sse = train(processed_tweets, k)

        # for every cluster c, printing size of each cluster
        for c in range(len(clusters)):
            print("---- size of cluster " + str(c+1) + " : ", str(len(clusters[c])) + " tweets")
            # printing tweets in a cluster
            # for t in range(len(clusters[c])):
            #     print("t" + str(t) + ", " + (" ".join(clusters[c][t][0])))

        print("---- SSE : " + str(sse))
        print('\n')

        # incrementing k after every experiment
        k = k+1