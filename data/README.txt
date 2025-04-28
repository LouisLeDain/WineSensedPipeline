wines_experiment_processed : 115 items, pas mal de manque
- experiment_id : clé primaire
- year : année de production du vin
- price : prix du vin (non disponible la plupart du temps)
- is_natural : t si le vin est naturel, c'est à dire sans intrant
- wine_alcohol : pourcentage d'alcool dans le vin
- region_country : deux lettres indiquant le pays de production

scraped_wines_processed : 95 items
- Vintage : année de production du vin
- Type : rouge, bland rosé etc …
- Producer : producteur du vin
- Variety : variété du vin 
- Designation : ramification d l'appelation, pas toujours nécessaire
- Vineyard : type de vigne (non disponible la plupart du temps)
- Country : pays de production du vin
- Region : région de production du vin
- SubRegion : détails de la région de production, pas toujours nécessaire
- Appelation : le nom trouvé sur CellarTracker
- experiment_id : clé primaire, permet de lier avec wines_experiment_processed
- wine : nom du vin tel qu'il était utilisé dans le dataset original

napping.csv : 2819 items
- experiment_id : même clé que dans les deux autres tables, permet de relier l'expérience à un vin
- pour les autres valeurs j'ai rien touché


Les images scrappées sont sous le format {experiment_id}.jpg
Je n'ai encore rencontré cette situation mais il n'est pas exclu que certaines images manquent