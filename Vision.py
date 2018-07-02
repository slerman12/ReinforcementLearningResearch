from __future__ import division
import cv2
import numpy as np
from skimage.measure import regionprops, label
from skimage.segmentation import felzenszwalb, mark_boundaries
from sklearn import random_projection
from sklearn.neighbors.kd_tree import KDTree
import matplotlib.pyplot as plt
import tensorflow as tf


class Vision:
    def __init__(self, size=None, greyscale=False, crop=None, params=None, brain=None):
        # State
        self.state = None

        # Pre-processing
        self.size = size
        self.greyscale = greyscale
        self.crop = crop

        # Parameters
        self.params = {} if params is None else params

        # Architecture ("brain module")
        self.brain = brain

        # Learning
        self.session = self.loss = self.train = self.accuracy = None

    def start_brain(self):
        pass

    def see(self, state):
        # Set state
        self.state = state

        # Greyscale
        if self.greyscale:
            self.state = np.dot(self.state[..., :3], [0.299, 0.587, 0.114])

        # Crop image
        if self.crop is not None:
            height = self.state.shape[0]
            width = self.state.shape[1]
            self.state = self.state[self.crop[0]:(height - self.crop[1]), self.crop[3]:(width - self.crop[2])]

        # Image resize
        if self.size is not None:
            self.state = cv2.resize(self.state, dsize=self.size)

        # Return meaningful representation
        return self.brain.run(inputs={"inputs": state})

    def experience(self, experience):
        pass

    def learn(self, inputs):
        # Train brain
        if self.train is not None:
            self.brain.run(inputs, self.train)


class Segmentation:
    def __init__(self, object_capacity, size=None, greyscale=False, crop=None, params=None, trajectory=True):
        # State
        self.state = None

        # Segments and their properties
        self.segments = None
        self.properties = None
        self.num_objects = 0

        # Prior segments
        self.prev_segments = None
        self.prev_properties = None
        self.prev_num_objects = 0
        self.linked = None

        # Objects in the current state
        self.scene = np.zeros((object_capacity, 5 if trajectory else 3))

        # Objects in the previous state
        self.prev_scene = np.zeros((object_capacity, 5 if trajectory else 3))

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

        # If trajectories
        self.trajectory = trajectory

    def see(self, state):
        # Set state
        self.state = state

        # Greyscale
        if self.greyscale:
            self.state = np.dot(self.state[..., :3], [0.299, 0.587, 0.114])

        # Crop image
        if self.crop is not None:
            height = self.state.shape[0]
            width = self.state.shape[1]
            self.state = self.state[self.crop[0]:(height - self.crop[1]), self.crop[3]:(width - self.crop[2])]

        # Image resize
        if self.size is not None:
            self.state = cv2.resize(self.state, dsize=self.size)

        # Set previous scene to current scene
        if self.num_objects > 0:
            self.prev_scene = self.scene.copy()
            # self.prev_segments = self.segments.copy()
            self.prev_properties = self.properties[:]
            self.prev_num_objects = self.num_objects

        # Segmentation
        self.segments = felzenszwalb(self.state, scale=self.params[0], sigma=self.params[1], min_size=self.params[2])

        # Properties for each segment
        self.properties = regionprops(self.segments)

        # Number of objects
        self.num_objects = len(self.properties)

        # If more segments than space
        if self.num_objects > self.object_capacity:
            # Use subset of objects
            print("\nExceeded object capacity: {} objects".format(self.num_objects))
            self.num_objects = self.object_capacity

        # Reset scene
        self.scene = np.zeros((self.object_capacity, 5 if self.trajectory else 3))

        # Add objects to scene
        for obj in range(self.num_objects):
            self.scene[obj, 0] = round(self.properties[obj].area)
            self.scene[obj, 1] = round(self.properties[obj].centroid[0])
            self.scene[obj, 2] = round(self.properties[obj].centroid[1])

        # Compute trajectories
        if self.trajectory:
            self.compute_trajectories()

        # print(self.scene)
        # print(self.prev_scene)
        # print()

        # Return flattened scene
        return self.scene.flatten()

    def learn(self):
        pass

    def compute_trajectories(self):
        # If there were previous objects
        if self.prev_num_objects > 0 and self.num_objects > 0:
            # Indices of previous objects that have been linked to and respective distances
            self.linked = np.full((self.num_objects, 2), -1)

            # Build KD tree for nearest neighbors on the previous scene TODO: can include trajectories
            tree = KDTree(self.prev_scene[:self.prev_num_objects, :-2], leaf_size=2)

            # Most similar objects
            dist, ind = tree.query(self.scene[:self.num_objects, :-2], k=1)

            # Uniquely link current objects to closest objects of the previous scene
            for obj in range(self.num_objects):
                # Set index and distance
                index = ind[obj][0]
                distance = dist[obj][0]

                # Check if already linked
                prior_link = np.where(self.linked[:, 0] == index)[0]

                # Link closest pair
                if prior_link.size > 0:
                    # Set prior link
                    prior_link = int(prior_link)

                    # If closer than prior pair
                    if distance < self.linked[prior_link, 1]:
                        # Unlink prior pair
                        self.linked[prior_link, 0] = -1
                        self.scene[prior_link, -2] = 0
                        self.scene[prior_link, -1] = 0

                        # Link this pair
                        self.linked[obj, 0] = index
                        self.linked[obj, 1] = distance
                        self.scene[obj, -2] = self.scene[obj, 1] - self.prev_scene[index, 1]
                        self.scene[obj, -1] = self.scene[obj, 2] - self.prev_scene[index, 2]
                else:
                    # Link this pair
                    self.linked[obj, 0] = index
                    self.linked[obj, 1] = distance
                    self.scene[obj, -2] = self.scene[obj, 1] - self.prev_scene[index, 1]
                    self.scene[obj, -1] = self.scene[obj, 2] - self.prev_scene[index, 2]

    def plot(self):
        # If state and segments have been set
        if self.state is not None and self.segments is not None:
            # Plot trajectories
            # print(self.segments[0])
            # print()
            linked_priors = np.zeros(self.segments.shape)
            i = 0
            for link in self.linked:
                if link[0] > -1:
                    # print(self.scene[i], self.prev_scene[link[0]])
                    for coord in self.prev_properties[link[0]].coords:
                        linked_priors[coord[0], coord[1]] = i + 1
                    i += 1
            linked_priors = label(linked_priors)

            # Show segments
            figure = plt.figure("Segments")
            figure.add_subplot(1, 1, 1)
            plt.imshow(mark_boundaries(self.state, linked_priors))
            # plt.axis("off")
            # figure.add_subplot(1, 2, 2)
            # plt.imshow(self.state)
            # plt.axis("off")
            # figure.add_subplot(1, 1, 1)
            # plt.imshow(mark_boundaries(self.state, self.segments))
            # plt.text(10, -2, 'Number of segments: {}, previous number: {}'.format(self.num_objects,
            #                                                                       self.prev_num_objects))
            # plt.text(10, -5, 'Previous number of segments: {}'.format(self.prev_num_objects))
            # plt.text(0, -2, '{} segments'.format(self.num_objects))
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

    def see(self, state):
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

    def learn(self, experience):
        pass
