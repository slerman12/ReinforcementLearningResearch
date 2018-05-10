import cv2
import numpy as np
from skimage.measure import regionprops
from skimage.segmentation import felzenszwalb, mark_boundaries
from sklearn import random_projection
from sklearn.neighbors.kd_tree import KDTree
import matplotlib.pyplot as plt


class Vision:
    # State
    state = None

    # Segments
    segments = None

    # Number of objects in the current state
    num_objects = 0

    # Number of objects in teh previous state
    prev_num_objects = 0

    def __init__(self, object_capacity, size=None, greyscale=False, crop=None, params=None):
        # Objects in the current state
        self.scene = np.zeros((object_capacity, 5))

        # Objects in the previous state
        self.prev_scene = np.zeros((object_capacity, 5))

        # Max number of objects
        self.object_capacity = object_capacity

        # Pre-processing
        self.size = size
        self.greyscale = greyscale
        self.crop = crop

        # Segmentation parameters
        if params is None:
            params = [3.0, 0.1, 1]
        self.params = params

    def see(self, state):
        # Set state
        self.state = state

        # Greyscale
        if self.greyscale:
            self.state = np.dot(self.state[..., :3], [0.299, 0.587, 0.114])

        # Crop top of image
        if self.crop is not None:
            height = self.state.shape[0]
            width = self.state.shape[1]
            self.state = self.state[self.crop[0]:(height - self.crop[1]), self.crop[3]:(width - self.crop[2])]

        # Image resize
        if self.size is not None:
            self.state = cv2.resize(self.state, dsize=self.size)

        # Set previous scene to current scene
        self.prev_scene = self.scene

        # Segmentation
        self.segments = felzenszwalb(self.state, scale=self.params[0], sigma=self.params[1], min_size=self.params[2])

        # Properties for each segment
        properties = regionprops(self.segments)

        # Number of objects
        self.num_objects = len(properties)

        # If more segments than space
        if self.num_objects > self.object_capacity:
            # Use subset of objects
            self.num_objects = self.object_capacity
            print("\nExceeded object capacity")

        # Add objects to scene
        for obj in range(self.num_objects):
            self.scene[obj, 0] = round(properties[obj].area)
            self.scene[obj, 1] = round(properties[obj].centroid[0])
            self.scene[obj, 2] = round(properties[obj].centroid[1])

        # Compute trajectories
        self.compute_trajectories()

        # Return flattened scene
        return self.scene.flatten()

    def compute_trajectories(self):
        # Indices of previous objects that have been linked to and respective distances
        linked = np.full((self.num_objects, 2), -1)

        # Build KD tree for nearest neighbors on the previous scene
        tree = KDTree(self.prev_scene[:, :-2], leaf_size=2)

        # Most similar objects
        dist, ind = tree.query(self.scene[:, :-2], k=1)

        # Uniquely link current objects to closest objects of the previous scene
        for obj in range(self.num_objects):
            # Check if already linked
            prior_link = np.where(linked[:, 0] == ind[obj])[0]

            # Link closest pair
            if prior_link.size > 0:
                # If closer than prior pair
                if dist < linked[prior_link, 1]:
                    # Unlink prior pair
                    linked[prior_link, 0] = -1
                    self.scene[prior_link, -2] = 0
                    self.scene[prior_link, -1] = 0

                    # Link this pair
                    linked[obj, 0] = ind[obj]
                    linked[obj, 1] = dist[obj]
                    self.scene[obj, -2] = self.scene[obj, 1] - self.prev_scene[ind[obj], 1]
                    self.scene[obj, -1] = self.scene[obj, 2] - self.prev_scene[ind[obj], 2]
            else:
                # Link this pair
                linked[obj, 0] = ind[obj]
                linked[obj, 1] = dist[obj]
                self.scene[obj, -2] = self.scene[obj, 1] - self.prev_scene[ind[obj], 1]
                self.scene[obj, -1] = self.scene[obj, 2] - self.prev_scene[ind[obj], 2]

    def plot(self):
        # If state and segments have been set
        if self.state is not None and self.segments is not None:
            # Show segments
            figure = plt.figure("Segments")
            ax = figure.add_subplot(1, 1, 1)
            ax.imshow(mark_boundaries(self.state, self.segments))
            plt.axis("off")

            # Plot
            plt.show()


# Lower dimension projection of state
class RandomProjection:
    # Initialize projection
    projection = None
    
    def __init__(self, dimension, flatten=False, size=None, greyscale=False, crop=None):
        # Initialize variables
        self.dimension = dimension
        self.size = size
        self.greyscale = greyscale
        self.crop = crop
        self.flatten = flatten

    def update(self, state):
        # Greyscale
        if self.greyscale:
            state = np.dot(state[..., :3], [0.299, 0.587, 0.114])

        # Crop top of image
        if self.crop is not None:
            height = state.shape[0]
            width = state.shape[1]
            state = state[self.crop[0]:(height - self.crop[1]), self.crop[3]:(width - self.crop[2])]

        # Image resize
        if self.size is not None:
            state = cv2.resize(state, dsize=self.size)

        # Flatten
        if self.flatten:
            state = state.flatten()

        # Lower dimension projection
        if self.projection is None:
            self.projection = random_projection.GaussianRandomProjection(self.dimension).fit([state])

        # Return embedded state
        return self.projection.transform([state])[0]
