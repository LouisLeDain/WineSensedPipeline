import pandas as pd

scraped_data = pd.read_csv('data/scraped_wines_processed.csv')

#creating global_id feature
scraped_data['global_id'] = scraped_data['experiment_id']

# renaming Vintage feature to year
scraped_data['year'] = scraped_data['Vintage']
scraped_data.drop(columns=['Vintage'], inplace=True)

# dropping desgination, subregion, wine features
scraped_data.drop(columns=['Designation', 'SubRegion', 'wine'], inplace=True)

# renaming Appellation feature to wine_name
scraped_data['wine_name'] = scraped_data['Appellation']
scraped_data.drop(columns=['Appellation'], inplace=True)

# adding alcohol feature
processed = pd.read_csv('data/wines_experiment_processed.csv')
complete_dataset = pd.merge(scraped_data, processed[['experiment_id', 'wine_alcohol']], on='experiment_id', how='left')

#save the updated file
complete_dataset.to_csv('data/updated_wine_processed.csv', index=False)

print('Wine dataset updated')