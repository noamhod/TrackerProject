#!/usr/bin/python
import os
import math
import array
import numpy as np
from collections import deque, defaultdict


from tracker_lib import config, utils
from tracker_lib.objects import Hit
from tracker_lib.objects import Cls

#############################################
### Simple pac-man recursion
def Pacman_Clustering(cluster_pixels,pivot,pixels):
    cfg = config.Config().map
    if(pivot in pixels):
        cluster_pixels.append(pivot) ## add this pixel to the cluster
        pixels.remove(pivot)         ## kill pixel from live pixels list
        for pixel in pixels[:]:
            dx = abs(pixel.x-pivot.x)
            dy = abs(pixel.y-pivot.y)
            adjacent_with_diagonals_allowed  = (cfg["allow_diagonals"] and ((dx+dy)<=2 and dx<=1 and dy<=1))
            adjacent_with_diagonals_forbiden = (not cfg["allow_diagonals"] and (dx+dy==1))
            # if((dx+dy)<=2 and dx<=1 and dy<=1): ## default
            # if(dx+dy==1): ## without diagonal connections
            if(adjacent_with_diagonals_allowed or adjacent_with_diagonals_forbiden):
                nextpivot = pixel
                RecursiveClustering(cluster_pixels,nextpivot,pixels)


def Pacman_GetAllClusters(pixels,det):
    clusters  = []
    positions = []
    CID = 0
    while len(pixels)>0:  ## loop as long as there are live pixels in the list
        pixel = pixels[0] ## this is the pivot pixel for the cluster recursion
        cluster_pixels = []
        Pacman_Clustering(cluster_pixels,pixel,pixels)
        cluster = Cls(cluster_pixels,det,CID)
        CID += 1
        clusters.append( cluster )
    return clusters


#############################################
### DFS
def DFS_Clustering(cluster_pixels, pivot, pixels):
    """Performs a depth-first search to find all connected pixels in a cluster."""
    # Convert pixels to a set for fast lookups
    pixels_set = {(pixel.x, pixel.y) for pixel in pixels}
    det = pixels[0].det
    stack = [pivot]

    while stack:
        current = stack.pop()
        if current in pixels_set:
            cluster_pixels.append(current)
            pixels_set.remove(current)  # Remove to avoid revisiting
            # Define direct neighbors (no diagonal connections)
            neighbors = [
                (current[0] + 1, current[1]), (current[0] - 1, current[1]),
                (current[0], current[1] + 1), (current[0], current[1] - 1)
            ]
            # Add valid neighbors to the stack
            for neighbor in neighbors:
                if neighbor in pixels_set:
                    stack.append(neighbor)

    # Update the original pixels list to only keep non-clustered pixels
    remaining_pixels = [Hit(det,x,y) for x, y in pixels_set]
    pixels.clear()
    pixels.extend(remaining_pixels)

def DFS_GetAllClusters(pixels,det):
    clusters = []
    CID = 0
    while pixels:
        # Use the first pixel in the list as the pivot
        pivot = (pixels[0].x, pixels[0].y)  # Convert to a tuple for consistency
        cluster_pixels = []
        # Find all pixels connected to pivot and add to cluster_pixels
        DFS_Clustering(cluster_pixels, pivot, pixels)
        # Convert back to Pixel objects for Cls if needed
        cluster_pixels = [Hit(det,x,y) for x, y in cluster_pixels]
        # Create and add the cluster to the clusters list
        cluster = Cls(det,cluster_pixels)
        CID += 1
        clusters.append(cluster)
    return clusters


#############################################
### BFS 
def BFS_Clustering(cluster_pixels, pivot, pixels_dict):
    ### Performs a breadth-first search (BFS) to find all connected pixels in a cluster
    queue = deque([pivot])
    while queue:
        current = queue.popleft()
        if current in pixels_dict:
            cluster_pixels.append(current)
            del pixels_dict[current]  # Remove pixel from the dictionary to mark as processed
            # Define direct neighbors (no diagonal connections)
            neighbors = [
                (current[0] + 1, current[1]), (current[0] - 1, current[1]),
                (current[0], current[1] + 1), (current[0], current[1] - 1)
            ]
            # Enqueue valid neighbors
            for neighbor in neighbors:
                if neighbor in pixels_dict:
                    queue.append(neighbor)

def BFS_GetAllClusters(pixels,det):
    cfg = config.Config().map
    clusters = []
    # Create a dictionary for fast pixel lookups
    pixels_dict = {(pixel.x, pixel.y): pixel for pixel in pixels}
    CID = 0
    while pixels_dict:
        # Start with an arbitrary pixel from the dictionary as the pivot
        pivot = next(iter(pixels_dict))
        pivot_pix = pixels_dict[pivot]
        cluster_pixels = []
        # Find all pixels connected to pivot using BFS
        BFS_Clustering(cluster_pixels, pivot, pixels_dict)
        # Convert back to Pixel objects for Cls if needed
        cluster_pixels = [pixels_dict.get((x,y), Hit(det,x,y)) for x,y in cluster_pixels]
        if(len(cluster_pixels)==1 and cfg["isFakeMC"]):
            cluster_pixels = [pivot_pix]
        # Create and add the cluster to the clusters list
        cluster = Cls(det,cluster_pixels,CID)
        # print(f"in BFS: xFake={cluster_pixels[0].xFake}, yFake={cluster_pixels[0].yFake}")
        # print(f"in BFS: x={cluster.xmm}, y={cluster.ymm}")
        CID += 1
        clusters.append(cluster)
    return clusters