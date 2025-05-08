# WineSensedPipeline

## RECENTLY ADDED

- Fichier config.py qui contient tous les paramètres (Les fonctions de processs et de test_process ont été modifiés en conséquences)
- Batch sur GPU
- Ajout du fichier data/experiment_id_reviews.csv contenant des reviews pour les vins ayant un experiment_id (scrapé par les chercheurs)
- Création dans process.py de la fonction compute_mean_review_embedding qui a partir du fichier csv renvoye l'embedding de texte moyen pour chaque experiment_id (on va enfin pouvoir implémenter FEAST)

## TO DO


•⁠  ⁠Implémenter le calcul des positions tsne sur le dataset global (avec une clé « global_id » imaginaire pour le moment)

•⁠  ⁠Modifier FEAST pour qu’il n’ait plus à calculer les machine kernel (les positions tsne)

•⁠  ⁠⁠Implémenter une fonction qui calcule les positions 2D finales, étant données les positions tsne et les poids obtenus par feast

•⁠  ⁠⁠Implémenter FEAST pour les reviews (…)

•⁠  ⁠⁠Implémenter FEAST V2 avec les caractéristiques

•⁠  ⁠⁠Implémenter les métriques pour tester FEAST (discussion sur la version V2 pour la notation)


## Data Architecture

data/

  ∟-> image/ = 'global_id'.jpg files
      
  ∟-> napping.csv = {session_round_name, event_name, experiment_no, experiment_id, coor1, coor2, color}
  
  ∟-> scraped_wines.csv = {global_id, year, type, producer, variety, vineyard, country, region, wine_name, alcohol, experiment_id (if used for human kernel experiment)}
  
  ∟-> reviews.csv = {global_id, review}
