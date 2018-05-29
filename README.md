## Autoencoding Beyond Pixels Using a Learned Similarity Metric (A Keras Implementation)
EE 298 Deep Learning Project
Jethro Racelis | Roxanne Avinante
May 28, 2018

Implementation of the method described in [Arxiv paper](https://arxiv.org/abs/1512.09300) using Keras.

Authors: Anders Boesen Lindbo Larsen, Søren Kaae Sønderby, Hugo Larochelle, Ole Winther

#### Implementation and Sample Results
Presentation Slides: https://bit.ly/2xq09XI

#### Dataset
http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
202,599 number of face images

#### Usage
* Training the model

python3 vaegan.py

* Generating images

python3 vaegan_use_trained_model.py


*Note: Results can be further improve if the model will be trained using GPU*
