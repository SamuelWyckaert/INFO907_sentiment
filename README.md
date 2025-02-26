# Détection de Propos Haineux avec un Modèle de Langage

## Introduction

Ce projet consiste à entraîner et déployer un modèle de classification de texte pour détecter les propos haineux en français. Il est basé sur `distilbert-base-multilingual-cased` et est affiné sur un jeu de données de détection de discours haineux.

## Structure du Projet

Le projet est composé de trois parties principales :

1. **Entraînement et Affinage du Modèle** : Utilisation de `transformers` et `datasets` pour entraîner un modèle sur un jeu de données annoté.
2. **Déploiement du Modèle avec FastAPI** : Un serveur FastAPI permettant d'exposer le modèle via une API REST.
3. **Interface Web en HTML & JavaScript** : Un front-end minimaliste permettant aux utilisateurs de tester la classification de leurs textes.

## Initialisation

- Création d'un environnement virtuel : `python3 -m venv env`

- lancement de l'environnement : `source venv/bin/activate`

- installation des requirements : `pip install -r requirement.txt`

- lancement du server : `uvicorn server:app --host 0.0.0.0 --port 8000`

---

## 1. Entraînement du Modèle

### Chargement et Prétraitement des Données

- On utilise le dataset `manueltonneau/french-hate-speech-superset` de Hugging Face.
- Le jeu de données est divisé en 5000 éléments pour l'entraînement et 2000 éléments pour la validation.

- Dans un premier temps, l'utilisation du modèle `distilbert-base-multilingual-cased` puis du modèle `almanach/camembertav2-base`.

- Les données sont formatées pour être compatibles avec le modèle de classification (`label` au lieu de `labels`).

### Entraînement

- Utilisation de `Trainer` de `transformers` avec les paramètres suivants :
  - `learning_rate=2e-5`
  - `per_device_train_batch_size=16`
  - `per_device_eval_batch_size=16`
  - `num_train_epochs=3`
  - `weight_decay=0.01`
- La métrique d'évaluation utilisée est `accuracy`.
- Le modèle entraîné est sauvegardé localement.

### Sauvegarde et Publication sur Hugging Face

- Le modèle et le tokenizer sont sauvegardés et poussés sur Hugging Face Hub sous le repository `theobalzeau/my-hate-speech-model`.

---

## 2. Déploiement avec FastAPI

### API avec FastAPI

- Un serveur FastAPI est mis en place pour exposer une API REST.
- Un middleware CORS est activé pour permettre les requêtes depuis un client web.
- L'API charge le modèle `theobalzeau/my-hate-speech-model2` et le met sur le `device` disponible (`MPS` ou `CPU`).
- L'endpoint `/test` reçoit un texte et retourne sa classification (`Message Haineux` ou `Message Non Haineux`).

---

## 3. Interface Web

### Structure

- Une page HTML simple avec un formulaire contenant un `textarea` et un bouton d'envoi.
- Un `fetch` en JavaScript permet d'envoyer le texte à l'API et d'afficher la réponse.

### Fonctionnement

- L'utilisateur entre un texte.
- Le texte est envoyé via une requête `POST` à `http://127.0.0.1:8000/test`.
- La réponse est affichée sous le formulaire.

---

## Tests et Résultats

## Remarque préliminaire

Nous tenons à préciser que les observations suivantes ne sont que le reflet des résultats obtenus par le modèle et ne constituent en aucun cas une prise de position ou un jugement personnel.

## Analyse des résultats

- Nous avons d'abord essayé de tuner le modèle `distilbert-base-multilingual-cased`, avec un dataset d'entraînement de 5000 et 2000 de test, mais nous avons remarqué un biais envers l'islam : chaque phrase contenant le mot "islam" était considérée comme haineuse, à l'exception notable de "J'aime l'islam". De plus, le modèle différenciait les minuscules des majuscules. Une phrase considérée comme haineuse en minuscules n'était pas haineuse en majuscules.

- Nous avons ensuite essayé de tuner le modèle `almanach/camembertav2-base`, mais les résultats étaient encore moins satisfaisants que le modèle précédent.

- Afin d'améliorer la performance, nous avons augmenté le dataset d'entraînement à 10 000 éléments et retravaillé avec `distilbert-base-multilingual-cased`. Cependant, les résultats se sont encore détériorés, avec un biais persistant envers l'islam. Même la phrase "J'aime l'islam" était classée comme haineuse.

## Conclusion

Ce projet met en lumière les difficultés liées à l'entraînement de modèles de classification de texte, notamment les biais présents dans les modèles pré-entraînés et la nécessité d'un dataset diversifié et équilibré.

Des améliorations potentielles pourraient inclure :

- L'utilisation d'un dataset plus large et mieux annoté.

- Un affinement des techniques de prétraitement des données.

---

## Technologies utilisées

- Python, Transformers (Hugging Face)
- FastAPI
- HTML, CSS, JavaScript
- Torch (avec support `MPS` pour MacBook M1/M2`)

## Participants

- Samuel Wyckaert
- Morgan Bazin
- Théo Balzeau
