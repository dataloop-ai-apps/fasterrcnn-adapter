# fasterrcnn-adapter
Model adapter for faster rcnn object and segmentation.

A Dataloop implementation of the [Faster-RCNN](https://arxiv.org/abs/1506.01497) architecture, following [torchvision's tutorial for FasterRCNN] (https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html).

## Pre-requisites
- torch
- torchvision
- numpy
- pycocotools

## How to install

Choose a Dataloop project and dataset and run the ```model_adapter.py``` script as follows:

```bash
python model_adapter.py -e <prod> -d <dataset_id> -p <project_id>
```

This will install the package and create the model in the appropriate project.


## Training/Finetuning

1. [Define the subsets](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#finetune-on-a-custom-dataset), related to the directories containing train and validation sets in the model's dataset
2. [Run a training process](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#train) on the instance


## Predict/Deploy

Follow these [instructions] on how to deploy and predict with the trained model.

## Short example of deploying torchvision's Faster-RCNN

Dataloop's documentation includes an [example](https://developers.dataloop.ai/tutorials/model_management/new_model_torchvision_example/chapter/#deploy-the-model) of how to quickly deploy and predict using Faster-RCNN
