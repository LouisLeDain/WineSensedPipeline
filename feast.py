from process import pairwise_distance_matrix, perform_nmds,perform_tsne, perform_clip_from_text, \
perform_clip_from_image, perform_clip_from_image_and_text, perform_cca,compute_experiment_ids_in_images, compute_mean_review_embedding
from config import MAX_REVIEW_SAMPLE_SIZE
from PIL import Image
import os
import re
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.cross_decomposition import CCA 
from sklearn.preprocessing import StandardScaler
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

    def compute_tar(self):
        
        # Compute the common experiment IDs
        common_ids = set(self.wine_id_to_index_human.keys())&(set(self.wine_id_to_index_machine.keys()))
        common_ids = sorted(list(common_ids))
        self.common_ids = common_ids

        
        print(f"Number of common experiment IDs: {len(common_ids)}")
        
        # Filter the distance matrix to only include common experiment IDs
        human_kernel_common = self.human_kernel[[self.wine_id_to_index_human[wine_id] for wine_id in common_ids]]
        machine_kernel_common = self.machine_kernel[[self.wine_id_to_index_machine[wine_id] for wine_id in common_ids]]
        

        
        idx = [self.wine_id_to_index_human[w] for w in common_ids]
        D_full = self.distance_matrix
        D = D_full[np.ix_(idx, idx)]
        N = len(common_ids)

        # 1. Build all triplets from your human-distance matrix D (NxN)
        triplets = [] # The index are only on the Human kernel
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    if D[i,j] > 0 and D[i,k] > 0 and D[i,j] < D[i,k]: # We put >0 because we don't want to have 0 distance (missing value)
                        triplets.append((i,j,k))
                        
        # 1b. Split so train/test triplets are disjoint
        all_wines = set(range(N))
        train_wines, test_wines = train_test_split(list(all_wines), test_size=0.2, random_state=42)
        test_triplets  = [(i,j,k) for (i,j,k) in triplets if i in test_wines  and j in test_wines  and k in test_wines ]

        # 2. Fit CCA on train wines only
        X_train = machine_kernel_common[train_wines]   # CLIP+t-SNE for train wines
        Y_train = human_kernel_common[train_wines]     # NMDS for train wines
        x_scaler  = StandardScaler().fit(X_train)
        y_scaler = StandardScaler().fit(Y_train)
        X_train_s = x_scaler.transform(X_train)
        Y_train_s = y_scaler.transform(Y_train)
        cca     = CCA(n_components=2).fit(X_train_s, Y_train_s)

        # 3. Project *all* wines into the shared space
        X_all_s   = x_scaler.transform(machine_kernel_common)       # shape (N, feats)
        X_all_cca, _ = cca.transform(X_all_s, human_kernel_common)  # shape (N, 2)

        # 4. Compute TAR on test triplets
        agree = 0
        for i, j, k in test_triplets:
            dij = np.linalg.norm(X_all_cca[i] - X_all_cca[j])
            dik = np.linalg.norm(X_all_cca[i] - X_all_cca[k])
            if dij < dik:
                agree += 1

        tar = agree / len(test_triplets)
        print(f"Triplet Agreement Ratio: {tar:.3f}")
        
        
        # Test part to compute the TAR on the human kernel alone
        tar_human = sum(
        np.linalg.norm(human_kernel_common[i] - human_kernel_common[j]) < np.linalg.norm(human_kernel_common[i] - human_kernel_common[k])
        for i, j, k in test_triplets
        ) / len(test_triplets)

        print("TAR (human NMDS alone):", tar_human)
