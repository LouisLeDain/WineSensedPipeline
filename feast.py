from process import pairwise_distance_matrix, perform_nmds,perform_tsne, perform_clip_from_text, \
perform_clip_from_image, perform_clip_from_image_and_text, perform_cca,compute_experiment_ids_in_images
from tsne import Tsne
from PIL import Image
import os
import re

class Feast(object):
    """
    Class to handle the Feast pipeline.
    """

    def __init__(self, path_to_img,napping_csv,scraped_csv,experiment_csv,text_model = None, image_model = None):
        """
        Initialize the Feast pipeline.

        Args:
            feast_config (dict): Configuration for the Feast pipeline.
        """
        self.path_to_img = path_to_img
        self.napping_csv = napping_csv
        self.text_model = text_model
        self.image_model = image_model
        self.scraped_csv = scraped_csv
        self.experiment_csv = experiment_csv
        self.distance_matrix = None
        self.experiments_ids = None
        self.global_ids = sorted(scraped_csv['global_ids'].unique())
        self.wine_id_to_index_human = None
        self.wine_id_to_index_machine = None
        self.distance_matrix = None
        self.human_kernel = None
        self.machine_kernel= None
        
    def compute_human_kernel(self):
        """
        Compute the human kernel embeddings.
        """
        # Compute the distance matrix
        print("Computing pairwise distance matrix...")
        self.distance_matrix, self.experiments_ids = pairwise_distance_matrix(self.napping_csv)

        # Create a mapping from wine IDs to matrix indices
        self.wine_id_to_index_human = {experiment_id.astype(int): idx for idx, experiment_id in enumerate(self.experiments_ids)}

        # Print some statistics
        print(f"Number of unique human tasted wines: {len(self.experiments_ids)}")

        # Create 2d points using the matrix
        print("Creating 2D points using the distance matrix...")
        self.human_kernel = perform_nmds(self.distance_matrix)
        
        print("Human kernel computed successfully.")
        
    def compute_machine_kernel(self):
        """
        Run the machine kernel.
        """
       tsne_compute = Tsne(path_to_img='data/images/', image_model='clip',text_model=None, scraped_csv='data/scraped.csv')
       tsne_compute.run()
       self.machine_kernel = tsne_compute.tsne_positions
       self.wine_id_to_index_machine = tsne_compute.global_id

            
    def run(self):
        """
        Run the Feast pipeline.
        """
        self.compute_human_kernel()
        self.compute_machine_kernel()
        
        # Remove wines that are in the human kernel but have no image
        valid_experiment_ids = compute_experiment_ids_in_images(self.path_to_img) # give us the ids of the wines that have images
        # Check for missing IDs in both index maps
        missing_human = [wid for wid in valid_experiment_ids if wid not in self.wine_id_to_index_human]
        missing_machine = [wid for wid in valid_experiment_ids if wid not in self.wine_id_to_index_machine]

        if missing_human or missing_machine:
            print(f"Missing from human kernel: {missing_human}")
            print(f"Missing from machine kernel: {missing_machine}")
            raise KeyError("Missing wine_ids in index maps.")

        self.human_kernel = self.human_kernel[[self.wine_id_to_index_human[wine_id] for wine_id in valid_experiment_ids]]
        self.machine_kernel = self.machine_kernel[[self.wine_id_to_index_machine[wine_id] for wine_id in valid_experiment_ids]]
        

        weights = perform_cca(self.human_kernel, self.machine_kernel)
        self.weights = weights
        print("Weights computed successfully.")        
        print("Feast pipeline completed successfully.")

if __name__ == "__main__":
    # Example usage
    feast = Feast(path_to_img='data/images/', napping_csv='data/napping.csv',image_model='clip',text_model=None, scraped_csv='data/scraped.csv',experiment_csv='data/experiment.csv')

    feast.run()
    print("Feast pipeline completed.")