import numpy as np
from sklearn.datasets import load_iris
import os
from PIL import Image
from process import perform_nmds, perform_tsne, perform_cca, perform_clip_from_text, perform_clip_from_image,compute_mean_review_embedding
from feast import Feast


from config import PATH_TO_IMG, NAPPING_CSV, IMAGE_MODEL, TEXT_MODEL, SCRAPED_CSV, EXPERIMENT_CSV, DEVICE,REVIEW_CSV, MAX_REVIEW_SAMPLE_SIZE
np.random.seed(0)

'''
Testing NMDS
'''

print("Testing NMDS")

distance_matrix = np.random.rand(5,5)

nmds_results = perform_nmds(distance_matrix)

print(np.shape(nmds_results))

print("NMDS test ran successfully")

'''
Testing t-SNE
'''

print("Testing t-SNE")

iris = load_iris()
X = iris.data

data_tsne = perform_tsne(X)

print(np.shape(data_tsne))

print("t-SNE test ran successfully")

'''
Testing CCA
'''

print("Testing CCA")

X1 = np.random.rand(100, 5)
X2 = np.random.rand(100, 5)

X2[:, 0] = X1[:, 0] + np.random.normal(0, 0.1, 100)
X2[:, 1] = X1[:, 1] + np.random.normal(0, 0.1, 100)

weights = perform_cca(X1,X2)

print(np.shape(weights))

print("CCA test ran successfully")

'''
Testing CLIP text
'''

print("Testing CLIP text")

text = ["a photo of a transformers logo", "a photo of a cat"]

text_embeddings = perform_clip_from_text(text,DEVICE)

print(np.shape(text_embeddings))

print("CLIP text test ran successfully")

'''
Testing CLIP Image
'''

print("Testing CLIP image")

image_folder = 'data/images'
image_file = '0.jpg'

image_path = os.path.join(image_folder, image_file)
image= Image.open(image_path)
image_embeddings = perform_clip_from_image(image,DEVICE)

print(np.shape(image_embeddings))

print("CLIP image test ran successfully")

print("testing Mean text embedding")
compute_mean_review_embedding( REVIEW_CSV, DEVICE,MAX_REVIEW_SAMPLE_SIZE)
print("start testing FEAST pipeline")

# Example usage to test the Feast pipeline
feast = Feast(PATH_TO_IMG, NAPPING_CSV,SCRAPED_CSV,EXPERIMENT_CSV,DEVICE,image_model=IMAGE_MODEL,text_model=TEXT_MODEL)

feast.run()
print("Feast pipeline completed.") 

