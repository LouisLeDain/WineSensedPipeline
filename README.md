# WineSensedPipeline


## TO DO

Faire un fichier config où l'on met les paramètres du type path, device, etc.

•⁠  ⁠Implémenter le calcul des positions tsne sur le dataset global (avec une clé « global_id » imaginaire pour le moment)
•⁠  ⁠Modifier FEAST pour qu’il n’ait plus à calculer les machine kernel (les positions tsne)
•⁠  ⁠⁠Implémenter une fonction qui calcule les positions 2D finales, étant données les positions tsne et les poids obtenus par feast
•⁠  ⁠⁠Implémenter FEAST pour les reviews (…)
•⁠  ⁠⁠Implémenter FEAST V2 avec les caractéristiques
•⁠  ⁠⁠Implémenter les métriques pour tester FEAST (discussion sur la version V2 pour la notation)
•⁠  ⁠⁠Faire fonctionner sur GPU avec batchs
