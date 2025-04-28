'''
Compute and store clip embeddings of the images and text elements.
'''
import json
import os 
from process import perform_nmds, perform_tsne, perform_cca, perform_clip_from_text, perform_clip_from_image

class CLIPImageEmbedding():
    def __init__(self):
        self.clip = perform_clip_from_image
    
    def compute_and_save_embedding(self, image_folder, image_files, save_path):
        embeddings = {}
        for image_file in image_files:
            image_path = os.path.join(image_folder, image_file)
            image_embeddings = perform_clip_from_image(image_path)
            embeddings[image_file] = image_embeddings.squeeze().tolist()
            
            print(f'Embedding of file {image_file} computed')

        output_file = os.path.join(image_folder, "image_embeddings.json")
        with open(output_file, 'w') as f:
            json.dump(embeddings_list, f)

        print(f"Embeddings saved to {output_file}")

image_embedding = CLIPImageEmbedding()
image_files = os.listdir('data/images/')
print(f'List of files is of size {len(image_files)} and contains : \n {image_files}')

image_embedding.compute_and_save_embedding(image_folder = 'data/images/', image_files = os.listdir('data/images/'), save_path = 'data/images/')
