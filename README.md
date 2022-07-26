# Vector-borne-satellite-predictor
*Deep learning for vector-borne diseases with satellite-imagery*.

**Sentinelhub grant**: Sponsoring request ID 1c081a: Towards a Smart Eco-epidemiological Model of Dengue in Colombia using Satellite in Collaboration with [MIT Critical Data Colombia](https://github.com/MITCriticalData-Colombia). 

<p align="left">
    <a href="https://www.python.org/">
      <img src="https://img.shields.io/badge/Python-3.8-ff69b4.svg" /></a>
    <a href= "https://pytorch.org/">
      <img src="https://img.shields.io/badge/PyTorch-1.8-2BAF2B.svg" /></a>
    <a href= "https://github.com/sebasmos/vector-borne-satellite-predictor/blob/main/LICENCE">
      <img src="https://img.shields.io/badge/License-MIT-blue.svg" /></a>
</p>
<hr/>

* Satellite imagery is extracted with [satellite.extractor](https://github.com/sebasmos/satellite.extractor).

* Find here the codes to generate the satellite features extracted used on the datathon [Make Health Latam Datathon](https://makehealthlatam.com/)

* Weights for models described here can be found in [MIT Critical Data Repository](https://github.com/MITCriticalData-Colombia/SatDengue_MakeHealth)

## Dataset

Download dataset as follows: 
```
from google.colab import auth
auth.authenticate_user()

# set your gcp project
!gcloud config set project mit-hst-dengue

!gsutil -q -m cp -r gs://colombia_sebasmos/DATASET_5_best_cities .

!ls DATASET_5_best_cities/
```

To download Dengue cases in Colombia - Tabular data:

```
!gdown --id 1RGrXHgvn60L4pHA40M0R0scszHLno5fD
!unzip "dengue.zip" -d .
!rm -f dengue.zip
```


## Notebooks

1. [Transfer-learning with Resnet50](https://github.com/sebasmos/vector-borne-satellite-predictor/blob/main/notebooks/Deep_learning_for_Vector_Borne_Diseases.ipynb): Feature extraction from satellite images from Colombia based on Resnet50 and PCA - collaboration in [1](https://github.com/MITCriticalData-Colombia).
1. [Transfer-Learning with Vision Transformers](https://github.com/sebasmos/vector-borne-satellite-predictor/blob/main/notebooks/VIT_Deep_learning_for_Vector_Borne_Diseases.ipynb): Feature extraction from satellite images from Colombia based on ViT models - collaboration in [1](https://github.com/MITCriticalData-Colombia).




## Copyright and License

Code released under the [MIT](https://github.com/sebasmos/vector-borne-satellite-predictor/blob/main/LICENCE) license.
