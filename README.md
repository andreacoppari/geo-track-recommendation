# Geodata to Mountain Track Descriptions

## Overview

A model that turns geodata into detailed mountain track descriptions. The model is built using finetuned versions of the phi-1.5 and Mistral-7b models. Later, embedding similarity is calculated to find a good match between user description and tracks description to create a recommendation system.

## Directory Structure

```
geodata-to-mountain-track-descriptions/
|-- code/
|   |-- finetuning_phi_1-5.ipynb
|   |-- finetuning_mistral_7b.ipynb
|   |-- ...
|-- data/
|   |-- datasets/
|   |   |-- dataset_eng.json
|   |   |-- export_track.geojson
|   |   |-- ...
|-- src/
|   |-- images/
|       |-- image_1.png
|       |-- image_2.png
|-- README.md
```

- **code**: This directory contains Python notebooks for the finetuning of the phi-1.5 and Mistral-7b GPT models. Each subdirectory corresponds to a specific model.

- **data**: This directory houses datasets and additional data-related resources. The `datasets` folder contains CSV files representing different datasets, while the `data_stuff` folder includes any supplementary data or preprocessing scripts.

- **src**: Here, you will find images that can be embedded in the README file to provide visual representations of the phi-1.5 and Mistral-7b GPT models.

## Model Description

The purpose of this repository is to showcase a model that utilizes geospatial data to generate detailed descriptions of mountain tracks. The finetuned phi-1.5 and Mistral-7b GPT models are at the core of this transformation process.

## Instructions for Use

1. Navigate to the `code` directory.
2. TODO

## Images

### First idea
![Phi-1.5 Model](src/images/image_1.png)

### Final idea
![Mistral-7b Model](src/images/image_2.png)