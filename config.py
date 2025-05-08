from torch.cuda import is_available
'''
This file contains the configuration settings for the project.
'''

PATH_TO_IMG = "data/images/"
NAPPING_CSV = "data/napping.csv"
IMAGE_MODEL = "clip"
TEXT_MODEL = None
SCRAPED_CSV="data/scraped.csv"
EXPERIMENT_CSV="data/experiment.csv"
DEVICE= "cuda" if is_available() else "cpu" # Use GPU if available, otherwise use CPU (For machine kernel)
