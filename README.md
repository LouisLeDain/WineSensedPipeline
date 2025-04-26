# WineSensedPipeline


## TO DO

### Bloc preprocessing data
Class dataset qui hérite de torch.utils.data.Dataset  (voir [ici](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html) ) comme ça on pourra faire un dataloader et facilement batcher

init :

Input :
- path datas ? (specifier combien de fichier on aura et qu'est-ce qu'il y a dedans)
- Quels feature on sélectionne (Images, métadonnée, Métadonnée + images ) ?

Output :
- rien, modifie dans une variable de classe la data qui pourra être appeler dans getitem


getitem: (ne pas oublier les _)

input:
- id

output:
- la data (dépend de ce qu'on a choisi avant)

### Bloc model 
init :

input:
- Quel modèle.s ont utilise et forme de la donnée ?

output:

- rien, on initialise juste une variable de classe pour le modèle

forward:

self explicit