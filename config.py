from torch.cuda import is_available
'''
This file contains the configuration settings for the project.
'''

PATH_TO_IMG = "data/images/"
NAPPING_CSV = "data/napping.csv"
IMAGE_MODEL = "clip" 
TEXT_MODEL = "clip"
SCRAPED_CSV="data/scraped.csv"
EXPERIMENT_CSV="data/experiment.csv"
REVIEW_CSV = "data/experiment_id_reviews.csv"
DEVICE= "cuda" if is_available() else "cpu" # Use GPU if available, otherwise use CPU (For machine kernel)
MAX_REVIEW_SAMPLE_SIZE = 1000 # Maximum number of reviews to sample for each experiment id