from __future__ import division, print_function
import argparse
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import random


def init_centroids(num_clusters, image):
    
    centroids = []
    chosenPixels = []
    for i in range(num_clusters):
        while True:
            r1 = random.randint(0,len(image)-1)
            r2 = random.randint(0,len(image)-1)
            if (r1,r2) not in chosenPixels:
                break
        chosenPixels.append((r1,r2))
        centroids.append(image[r1][r2])
    centroids_init = np.array(centroids)
    #print("CENTROIDS INIT: ",centroids_init)
    # *** END YOUR CODE ***

    return centroids_init


def update_centroids(centroids, image, max_iter=30, print_every=10):
    """
    Carry out k-means centroid update step `max_iter` times
    Parameters
    ----------
    centroids : nparray
        The centroids stored as an nparray
    image : nparray
        (H, W, C) image represented as an nparray
    max_iter : int
        Number of iterations to run
    print_every : int
        Frequency of status update

    Returns
    -------
    new_centroids : nparray
        Updated centroids
    """
    old_centroids = np.array([])
    new_centroids = centroids
    for i in range(max_iter):
        ###############################
        #assign pixels to centroids
        assignments = {}
        for px in range(len(image)):
            for px2 in range(len(image)):
                minDistance = float('inf')
                assignedCentroid = []
                for centroid in new_centroids:
                    distance = np.linalg.norm(image[px][px2] - centroid)
                    if distance < minDistance:
                        assignedCentroid = centroid
                        minDistance = distance
                tupCentroid = tuple(assignedCentroid) 
                if tupCentroid not in list(assignments.keys()):
                    assignments[tupCentroid] = [image[px][px2]]
                else:
                    assignments[tupCentroid].append(image[px][px2])
        #######################
        #find new centroids:
        new_centroids = []
        for centroid in list(assignments.keys()):
            np_centroid = np.array(assignments[centroid])
            new_centroid = np.sum(np_centroid,axis=0)/len(np_centroid)
            new_centroids.append(new_centroid)

        new_centroids = np.array(new_centroids)
        #print("ITERATION ",i," NEW CENTROID: ",new_centroids)
        if i % print_every ==0:
            print("NEW CENTROIDS: ",new_centroids)
        if i != 0:
            if np.sum(new_centroids -old_centroids) == 0:
                print("FINISHED IN: ",i," ITERATIONS")
                return new_centroids
        old_centroids = new_centroids

        print("FINISHED ITERATION: ",i)
    # *** END YOUR CODE ***
    print("FINAL CENTROIDS: ", new_centroids)
    return new_centroids


def update_image(image, centroids):

    """
    Update RGB values of pixels in `image` by finding
    the closest among the `centroids`

    Parameters
    ----------
    image : nparray
        (H, W, C) image represented as an nparray
    centroids : int
        The centroids stored as an nparray

    Returns
    -------
    image : nparray
        Updated image
    """
    for px in range(len(image)):
        for px2 in range(len(image)):
            minDistance = float('inf')
            assignedCentroid = []
            for centroid in centroids:
                distance = np.linalg.norm(image[px][px2] - centroid)
                if distance < minDistance:
                    assignedCentroid = centroid
                    minDistance = distance
            image[px][px2] = np.array(assignedCentroid)

    # *** END YOUR CODE ***

    return image


def main(args):
    print("SETTING UP")
    # Setup
    max_iter = args.max_iter
    print_every = args.print_every
    image_path_small = args.small_path
    image_path_large = args.large_path
    num_clusters = args.num_clusters
    figure_idx = 0
    print("VARIABLES Initialized")
    # Load small image
    image = np.copy(mpimg.imread(image_path_large))
    print('[INFO] Loaded small image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original small image')
    plt.axis('off')
    savepath = os.path.join('.', 'orig_small.png')
    plt.savefig(savepath, transparent=True, format='png', bbox_inches='tight')

    # Initialize centroids
    print('[INFO] Centroids initialized')
    centroids_init = init_centroids(num_clusters, image)

    # Update centroids
    print(25 * '=')
    print('Updating centroids ...')
    print(25 * '=')
    centroids = update_centroids(centroids_init, image, max_iter, print_every)

    # Load large image
    image = np.copy(mpimg.imread(image_path_large))
    image.setflags(write=1)
    print('[INFO] Loaded large image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original large image')
    plt.axis('off')
    savepath = os.path.join('.', 'orig_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    # Update large image with centroids calculated on small image
    print(25 * '=')
    print('Updating large image ...')
    print(25 * '=')
    image_clustered = update_image(image, centroids)

    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image_clustered)
    plt.title('Updated large image')
    plt.axis('off')
    savepath = os.path.join('.', 'updated_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    print('\nCOMPLETE')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--small_path', default='./peppers-small.tiff',
                        help='Path to small image')
    parser.add_argument('--large_path', default='./peppers-large.tiff',
                        help='Path to large image')
    parser.add_argument('--max_iter', type=int, default=150,
                        help='Maximum number of iterations')
    parser.add_argument('--num_clusters', type=int, default=16,
                        help='Number of centroids/clusters')
    parser.add_argument('--print_every', type=int, default=10,
                        help='Iteration print frequency')
    args = parser.parse_args()
    print("EXECUTING MAIN")
    main(args)
