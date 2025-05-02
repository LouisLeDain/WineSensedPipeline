import numpy as np 
from sklearn.manifold import MDS, TSNE
from sklearn.cross_decomposition import CCA 
from sklearn.preprocessing import StandardScaler
from PIL import Image
import pandas as pd
import torch
from transformers import CLIPModel, CLIPTokenizer, CLIPProcessor, CLIPImageProcessor
import re
import os
np.random.seed(0)



def perform_nmds(distance_matrix):
    '''
    Perform NMDS over data :
        input -> matrix where position (i,j) corresponds to the distance between elements i and j
        output -> 2D position (x,y) for every element i
    '''

    # Reshape the distance_matrix
    distance_matrix_symmetric = (distance_matrix + distance_matrix.T) / 2 # Ensure the symmetry of distances
    np.fill_diagonal(distance_matrix_symmetric, 0) # Ensure each point is at distance 0 of itself

    # Initialize and fit NMDS model
    nmds = MDS(n_components=2, metric=False, dissimilarity='precomputed', random_state=0)
    nmds_results = nmds.fit_transform(distance_matrix_symmetric) # (x_pos, y_pos)
    
    return nmds_results 


def perform_tsne(data):
    '''
    Perform t-SNE over data :
        input -> data with attributes
        output -> 2D position (x,y) for every element i of data
    '''

    # Rescaling data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    data_tsne = tsne.fit_transform(data_scaled) # (x_pos, y-pos)

    return data_tsne


def perform_cca(data1, data2):
    '''
    Perform the CCA and compute the weights for the machine_kernel data to compute the corrsponding projected coordinates :
        input -> ( data1 = machine_kernel data (after CLIP, t-SNE), data2 = human_kernel data (after NMDS) )
        output -> machine_kernel features weights
    '''

    # Rescaling data
    scaler1 = StandardScaler()
    scaler2 = StandardScaler()
    data1_scaled = scaler1.fit_transform(data1)
    data2_scaled = scaler2.fit_transform(data2)

    # Perform CCA
    cca = CCA(n_components=2)
    data1_cca, data2_cca = cca.fit_transform(data1_scaled, data2_scaled)
    weights = cca.x_loadings_ # (features, n_components)

    return weights 


def perform_clip_from_text(text):
    '''
    Compute CLIP embeddings over tabular data (text):
        input -> tabular data (table of text data)
        output -> CLIP embeddings
    '''

    # Setting up the model
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    tokenizer = CLIPTokenizer.from_pretrained(model_name)

    # Get text embeddings
    text_input = tokenizer(text, padding=True, return_tensors="pt")
    with torch.no_grad():
        text_embeddings = model.get_text_features(**text_input)
    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True) # (len(text), 512)

    return text_embeddings


def perform_clip_from_image(image):
    
    '''
    Compute CLIP embeddings over images:
        input -> image
        output -> CLIP embedding
    '''
    
    # Setting up the model
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    image_processor = CLIPImageProcessor.from_pretrained(model_name)

    # Get text embeddings
    image_input = image_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_embeddings = model.get_image_features(**image_input) ############## Tout se fait sur le CPU la ? à changer.
    image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True) # (n_img, 512)

    return image_embeddings
def perform_clip_from_image_and_text(image, text):
    '''
    Compute CLIP embeddings over images and text:
        input -> image
        output -> CLIP embedding
    '''
    
    # Setting up the model
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    # Get image embeddings 
    inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    text_embeds  = outputs.text_embeds    # shape: (batch_size, 1024)
    image_embeds = outputs.image_embeds   # shape: (batch_size, 1024)
    
    joint_embeds = (text_embeds + image_embeds)/2 # shape: (batch_size, 1024)
    
    return joint_embeds
    
def pairwise_distance_matrix(napping_csv):
    '''
    Compute the pairwise distance matrix of the data using the napping.csv file :
        input -> napping.csv
        output -> distance matrix
    '''
    

    
    # Read the data
    data = pd.read_csv(napping_csv)
    
    # Get unique experiment IDs
    unique_wine_ids = sorted(data['experiment_id'].unique())
    n_wines = len(unique_wine_ids)
    
    # Create a mapping from wine IDs to matrix indices
    id_to_index = {wine_id: idx for idx, wine_id in enumerate(unique_wine_ids)} # remember that the order of the wine ids is important for the distance matrix
    
    # Initialize distance matrix and count matrix
    distance_matrix = np.zeros((n_wines, n_wines))
    count_matrix = np.zeros((n_wines, n_wines))
    
    # Group data by session, event, and experiment number to get different napping sessions
    grouped = data.groupby(['session_round_name', 'event_name', 'experiment_no'])
    
    # Process each group
    for _, group in grouped:
        # For each pair of wines in this group, compute Euclidean distance
        for i, row1 in group.iterrows():
            wine_id1 = row1['experiment_id']
            coords1 = np.array([row1['coor1'], row1['coor2']])
            
            for j, row2 in group.iterrows():
                if i != j:  # Don't compute distance to itself
                    wine_id2 = row2['experiment_id']
                    coords2 = np.array([row2['coor1'], row2['coor2']])
                    
                    # Compute Euclidean distance
                    dist = np.linalg.norm(coords1 - coords2)
                    
                    # Update distance and count matrices
                    idx1, idx2 = id_to_index[wine_id1], id_to_index[wine_id2]
                    distance_matrix[idx1, idx2] += dist
                    count_matrix[idx1, idx2] += 1
    
    # Compute average distances (avoid division by zero)
    mask = count_matrix > 0
    distance_matrix[mask] = distance_matrix[mask] / count_matrix[mask]
    
    # For pairs that don't have a distance (count = 0), set to zero (mean not defined, not zero distance)
    distance_matrix[~mask] = 0
    
    # Make the matrix symmetric by averaging with its transpose
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    
    # Set diagonal to zero (distance from a wine to itself is zero)
    np.fill_diagonal(distance_matrix, 0)
    
    return distance_matrix, unique_wine_ids


def process_text_data(text_data):
    '''
    Process the text data before the CLIP embeddings
        input -> text data
        output -> processed text_data
    '''
    # Convert to lowercase
    text_data = [text.casefold() if isinstance(text, str) else text for text in text_data]
    
    # Remove punctuation
    text_data = [re.sub(r'[^\w\s]', '', text) if isinstance(text, str) else text for text in text_data]
    
    return text_data

def compute_experiment_ids_in_images(path_to_img):
    '''
    Compute the experiment ids in the images
        input -> path to the images
        output -> list of experiment ids
    '''
    # Get all image files in the directory
    image_files = os.listdir(path_to_img)
    
    # Extract experiment IDs from filenames
    experiment_ids = [re.search(r'(\d+)', file).group(1) for file in image_files if re.search(r'(\d+)', file)]
    
    return experiment_ids