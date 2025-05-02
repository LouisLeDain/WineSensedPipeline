'''
Compute and store clip embeddings of the images and text elements.
'''
import json
import os 
from process import perform_nmds, perform_tsne, perform_cca, perform_clip_from_text, perform_clip_from_image
from torch import save, load 
class CLIPImageEmbedding():
    def __init__(self):
        self.clip = perform_clip_from_image
    
    def compute_and_save_embedding(self, image_folder, image_files, save_path):
        embeddings = {}
        for image_file in image_files:
            image_path = os.path.join(image_folder, image_file) # Ce serait pas mieux d'utiliser directement les id de vins ?
            image_embeddings = perform_clip_from_image(image_path)
            embeddings[image_file] = image_embeddings.squeeze().tolist() # pourquoi on convertit en liste ? le but c'est d'avoir des tenseurs pour calculer vite ?
            
            print(f'Embedding of file {image_file} computed')

        output_file = os.path.join(save_path, "image_embeddings.pt")
        save(embeddings, output_file)


        print(f"Embeddings saved to {output_file}")
        
    def load_images_embeddings(self,path_to_embeddings):
        embeddings = load(path_to_embeddings)
        print(f"Embeddings loaded from {path_to_embeddings}")
        return embeddings
    
if __name__ == "__main__":
    # Example usage
    # Assuming you have a folder 'data/images/' with images to process
    # and you want to save the embeddings in the same folder.
    
    # Initialize the CLIPImageEmbedding class and compute embeddings
    image_embedding = CLIPImageEmbedding()
    image_files = os.listdir('data/images/')
    print(f'List of files is of size {len(image_files)} and contains : \n {image_files}')

    image_embedding.compute_and_save_embedding(image_folder = 'data/images/', image_files = os.listdir('data/images/'), save_path = 'data/')
    embeddings = image_embedding.load_images_embeddings('data/image_embeddings.pt') # embeddings is a dictionary where the keys are the name of the files
    print(f'Embeddings are of size {len(embeddings)}')
    print(f'Embeddings are of size {embeddings[image_files[0]]}')
