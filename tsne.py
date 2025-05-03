from process import pairwise_distance_matrix, perform_nmds,perform_tsne, perform_clip_from_text, \
perform_clip_from_image, perform_clip_from_image_and_text, perform_cca,compute_experiment_ids_in_images
from PIL import Image
import os
import re

class Tsne(object):
    """
    Class to compute t-sne machine kernel positions
    """

    def __init__(self, path_to_img, scraped_csv, text_model = None, image_model = None):
        self.path_to_img = path_to_img
        self.dataset = scraped_csv
        self.text_model = text_model
        self.image_model = image_model
        self.global_id = None
        self.tsne_positions = None
    
    def compute_machine_kernel(self):
        """
        Run the machine kernel.
        """
        if self.text_model is None and self.image_model is None:
            raise ValueError("Text and image models must be provided to compute the machine kernel.")
        
        if self.text_model is None:
            print("Computing image embeddings...")
            images = [Image.open(os.path.join(self.path_to_img, img)) for img in os.listdir(self.path_to_img)]
            if self.image_model == "clip":
                self.tsne_positions = perform_tsne(perform_clip_from_image(images))
                self.global_id = {int(re.search(r'(\d+)', img).group(1)): idx for idx, img in enumerate(os.listdir(self.path_to_img))} # we only take the global id and not the name of the file
            else:
                raise ValueError("Unsupported image model. Please provide a valid model.")
            
        elif self.image_model is None:
           pass
           ### TODO: Implement text model embedding computation
           # We need reviews to do that
    
    def run(self):
        self.compute_machine_kernel()

if __name__ == "__main__":
    # Example usage
    tsne = Tsne(path_to_img='data/images/', scraped_csv='data/scraped.csv', text_model=None, image_model='clip')
    tsne.run()
    print("Tsne coordinates computed.")