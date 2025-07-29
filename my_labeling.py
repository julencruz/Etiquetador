__authors__ = ['1667150', '1667663', '1565785']
__group__ = ''

from utils_data import read_dataset, read_extended_dataset, crop_images, visualize_retrieval
import Kmeans


if __name__ == '__main__':

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json')

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)

    # You can start coding your functions here
    
def retrieval_by_color(list_imagenes, Etiquetas_colores, query):
    if not isinstance(query, list):
        query = [query]
        
    Lista_final = []
    #miramos para cada imagen si hay la etiqueta de color especificada
    for index, imagen in enumerate(Etiquetas_colores):
        if all(elem in imagen for elem in query):
            Lista_final.append(list_imagenes[index])
            
    return Lista_final


def retrieval_by_shape(list_imagenes, Etiquetas_forma, query):
    Lista_final = []
    #miramos para cada imagen si hay la etiqueta de forma especificada
    for index, image in enumerate(Etiquetas_forma):
        if all(elem in image for elem in query):
            Lista_final.append(list_imagenes[index])
    return Lista_final


def retrieval_combined(list_imagenes, Etiqueta_color, Etiquetas_forma, queryC, queryF):
    if not isinstance(queryC, list):
        queryC = [queryC]  
    if not isinstance(queryF, list):
        queryF = [queryF]
        
    Lista_final = []
    #miramos para cada imagen si hay la etiqueta de forma y de color especificada
    for index, (img_color, img_forma) in enumerate(zip(Etiqueta_color, Etiquetas_forma)):
        if all(elem in img_color for elem in queryC) and all(elem in img_forma for elem in queryF):
            Lista_final.append(list_imagenes[index])
    return Lista_final



