
from KmeansImproved import __authors__, __group__, KMeans, distance, get_colors
from PIL import Image
import numpy as np
import utils_data
import time

test_folder = 'testCases_img/'

def read_image(image_filename):
    image_path = test_folder + image_filename
    image = Image.open(image_path)
    image_array = np.array(image)
    return image_array

def test_kmeans_on_image(image_filename, K, option):
    image_data = read_image(image_filename)
    kmeans = KMeans(image_data, K, {'km_init': option})
    kmeans.fit()
    utils_data.visualize_k_means(kmeans, image_data.shape)
    # utils_data.Plot3DCloud(kmeans)
    
    return kmeans.centroids

def test_DCW(image_filename, K, option):
    image_data = read_image(image_filename)
    kmeans = KMeans(image_data, K, {'km_init': option})
    kmeans.fit()
    return kmeans.withinClassDistance()

def test_colors(image_filename, K, option):
    image_data = read_image(image_filename)
    kmeans = KMeans(image_data, K, {'km_init': option})
    kmeans.fit()
    return get_colors(kmeans.centroids)

def test_best_k(image_filename, K, option, max_k):
    image_data = read_image(image_filename)
    kmeans = KMeans(image_data, K, {'km_init': option})    
    
    kmeans.find_bestK(max_k)
    
    return kmeans.K


# TEST 1: Question 2 Wooclap
print("Le has dado")

imagen = ['1529.jpg', '58182.jpg', 'img1_tw.jpg', 'img2_tw.jpg', '1845.jpg']

# test_1 = test_kmeans_on_image(imagen[2], 4, 'kmeans++')
k = test_best_k(imagen[2], 1, 'random', 10)
print("Test 1 Centroids:", "\n", k, "\n")
test_1 = test_kmeans_on_image(imagen[2], k, 'random')

#Correcta: [[192.00 192.02 128.05]
 	#      [192.96  64.03  127.96]
    #      [ 63.04 191.96 128.04]
 	#      [ 64.00  63.99 127.96]]

# TEST 2: Question 3 Wooclap
# test_2 = test_kmeans_on_image('img2_tw.jpg', 1, 'random')
# print("Test 2 Centroids:", test_2, "\n")
# #Correcta: [[128.00  128.00   128.00]]

# # TEST 3: Question 4 Wooclap
# test_3 = test_DCW('img3_tw.jpg', 2, 'first')
# print("Test 3 DCW:", test_3, "\n")
# #Correcta: WCD: 4960.64

# # TEST 4: Question 5 Wooclap
# test_4 = test_DCW('img2_tw.jpg', 1, 'first')
# print("Test 4 DCW:", test_4, "\n")
# #Correcta: WCD: 0.0. Because in a monochrome image, all points are assigned to the same cluster and are at the same distance from the centroid.

# # TEST 5: Question 6 Wooclap
# test_5 = test_colors('img1_tw.jpg', 4, 'first')
# print("Test 5 colors:", test_5, "\n")
# #Correcta: ['Yellow', 'Pink', 'Green', 'Blue']

# # TEST 6: Question 7 Wooclaps
# max_k = 10
# test_6 = test_best_k('img1_tw.jpg', 1, 'first', max_k)
# print("Test 6 best K:", test_6, "\n")
# #Correcta: 4