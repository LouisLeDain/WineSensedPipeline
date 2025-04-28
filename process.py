import numpy as np 
from sklearn.manifold import MDS, TSNE
from sklearn.cross_decomposition import CCA 
from sklearn.preprocessing import StandardScaler

import torch
from transformers import CLIPModel, CLIPTokenizer, CLIPProcessor, CLIPImageProcessor

np.random.seed(0)



def perform_nmds(distance_matrix):
    '''
    Perform NMDS over data :
        input -> matrix where position (i,j) corresponds to the distance between elements i and j
        output -> 2D position (x,y) for every element i
    '''

    # Reshape the distance_matrix
    disance_matrix_symmetric = (distance_matrix + distance_matrix.T) / 2 # Ensure the symmetry of distances
    np.fill_diagonal(distance_matrix_symmetric, 0) # Ensure each point is at distance 0 of itself

    # Initialize and fit NMDS model
    nmds = MDS(n_components=2, metric=False, dissimilarity='euclidean', random_state=0)
    nmds_results = nmds.fit_transform(distance_matrix) # (x_pos, y_pos)
    
    return nmds_results 


def perform_tSNE(data):
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
    processor = CLIPProcessor.from_pretrained(model_name)

    # Get text embeddings
    text_input = tokenizer(text, padding=True, return_tensors="pt")
    text_embeddings = model.get_text_features(**text_input)
    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

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
    img = Image.open(image)    
    image_input = image_processor(images=img, return_tensors="pt")
    image_embeddings = model.get_image_features(**image_input)
    image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)

    return image_embeddings