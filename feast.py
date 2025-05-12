from process import pairwise_distance_matrix, perform_nmds,perform_tsne, perform_clip_from_text, \
perform_clip_from_image, perform_clip_from_image_and_text, perform_cca,compute_experiment_ids_in_images, compute_mean_review_embedding
from config import MAX_REVIEW_SAMPLE_SIZE
from PIL import Image
import os
import re
import numpy as np
class Feast(object):
    """
    Class to handle the Feast pipeline.
    """

    def __init__(self, path_to_img,napping_csv,scraped_csv,experiment_csv,review_csv,device,text_model = None, image_model = None):
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
        self.review_csv = review_csv
        self.device = device
        self.distance_matrix = None
        self.wine_ids = None
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
        self.distance_matrix, self.wine_ids = pairwise_distance_matrix(self.napping_csv)

        # Create a mapping from wine IDs to matrix indices
        self.wine_id_to_index_human = {wine_id.astype(int): idx for idx, wine_id in enumerate(self.wine_ids)}

        # Print some statistics
        print(f"Number of unique wines: {len(self.wine_ids)}")

        # Create 2d points using the matrix
        print("Creating 2D points using the distance matrix...")
        self.human_kernel = perform_nmds(self.distance_matrix)
        
        print("Human kernel computed successfully.")
        
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
                self.machine_kernel = perform_tsne(perform_clip_from_image(images,self.device).cpu())
                self.wine_id_to_index_machine = {int(re.search(r'(\d+)', img).group(1)): idx for idx, img in enumerate(os.listdir(self.path_to_img))} # we only take the experiment id and not the name of the file
            else:
                raise ValueError("Unsupported image model. Please provide a valid model.")
            
        elif self.image_model is None:
            print("Computing text embeddings...") 
            if self.text_model == "clip":
                mean_embeddings_dict = compute_mean_review_embedding(self.review_csv,self.device,MAX_REVIEW_SAMPLE_SIZE)
                keys,values = zip(*mean_embeddings_dict.items())
                self.machine_kernel = perform_tsne(values)
                self.wine_id_to_index_machine = { key: idx for idx, key in enumerate(keys) }
            else:
                raise ValueError("Unsupported text model. Please provide a valid model.")
        
        else:
            if self.text_model == "clip" and self.image_model == "clip":
                print("Computing image and text embeddings...")
                images = [Image.open(os.path.join(self.path_to_img, img)) for img in os.listdir(self.path_to_img)]
                image_embeddings = perform_clip_from_image(images,self.device).cpu()
                experiment_id_to_image =  {int(re.search(r'(\d+)', img).group(1)): idx for idx, img in enumerate(os.listdir(self.path_to_img))}
                mean_embeddings_dict = compute_mean_review_embedding(self.review_csv,self.device,MAX_REVIEW_SAMPLE_SIZE)
                keys,text_embeddings = zip(*mean_embeddings_dict.items())
                experiment_id_to_text = { key: idx for idx, key in enumerate(keys) }
                
                common_ids= set(experiment_id_to_image.keys())&set(experiment_id_to_text.keys())
                common_ids = sorted(list(common_ids))
                
                image_embeddings = image_embeddings[[experiment_id_to_image[wine_id] for wine_id in common_ids]]
                text_embeddings = np.array(text_embeddings)
                text_embeddings = text_embeddings[[experiment_id_to_text[wine_id] for wine_id in common_ids]]
                self.wine_id_to_index_machine = {wine_id: idx for idx, wine_id in enumerate(common_ids)}
                combined_embeddings = (image_embeddings + text_embeddings)/2
                self.machine_kernel = perform_tsne(combined_embeddings)
                print("Machine kernel computed successfully.")
                
    def run(self):
        """
        Run the Feast pipeline.
        """
        self.compute_human_kernel()
        self.compute_machine_kernel()
        
        # Compute the common experiment IDs
        common_ids = set(self.wine_id_to_index_human.keys())&(set(self.wine_id_to_index_machine.keys()))
        common_ids = sorted(list(common_ids))
        self.common_ids = common_ids
        print(f"Number of common experiment IDs: {len(common_ids)}")
        
        # Filter the distance matrix to only include common experiment IDs
        human_kernel_common = self.human_kernel[[self.wine_id_to_index_human[wine_id] for wine_id in common_ids]]
        machine_kernel_common = self.machine_kernel[[self.wine_id_to_index_machine[wine_id] for wine_id in common_ids]]

        # Compute the CCA weights
        self.human_kernel_aligned, self.machine_kernel_aligned = perform_cca(human_kernel_common, machine_kernel_common)
        print("Weights computed successfully.")        
        print("Feast pipeline completed successfully.")

    def inference(self):
        pass