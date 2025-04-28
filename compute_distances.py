import os
import json
import pandas as pd
import numpy as np

data = pd.read_csv('data/napping.csv')
wines = pd.read_csv('data/wines_experiment_processed.csv')

n_wines = wines['experiment_id'].max()
distances = np.zeros((n_wines, n_wines, 2))

'''
Dans 'napping.csv'
experiment_id -> index 3
coor1 -> 4
coor2 -> 5
'''

for i in range(len(data)//5):
    for j in range(0, 4):
        for k in range(0, 4): 
            distances[data.iloc[i+j][3]-1, data.iloc[i+k][3]-1, 0] += 1
                
            vec1 = np.array(data.iloc[i+j][4], data.iloc[i+j][5])
            vec2 = np.array(data.iloc[i+k][4], data.iloc[i+k][5])
            d = np.linalg.norm(vec1-vec2)

            distances[data.iloc[i+j][3]-1, data.iloc[i+k][3]-1, 0] += d

final_distances = np.full((n_wines, n_wines), np.inf)

for i in range(n_wines):
    for j in range(n_wines):
        if distances[i,j,1] != 0:
            final_distances[i,j] = distances[i,j,0]/distances[i,j,1]

distance_matrix = {'distance_matrix' : final_distances.tolist}
output_file = os.path.join('data/', "distances.json")

with open(output_file, 'w') as f:
            json.dump(distance_matrix, f)


print(f"Distances saved to {output_file}")

