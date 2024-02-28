# Faster RCNN Model Adapter

## Introduction

This repo is a model integration between the [Pytorch implementation of FasterRCNN](http://pytorch.org/vision/master/models/faster_rcnn.html) segmentation model and [Dataloop](https://dataloop.ai/)

The Faster R-CNN model, an enhancement of Fast R-CNN, incorporates a region proposal network (RPN) alongside the CNN model. Unlike Fast R-CNN, it efficiently utilizes the RPN to share full-image convolutional features with the detection network, allowing for nearly cost-free region proposals. Operating as a fully convolutional network, the RPN simultaneously predicts object bounds and objectness scores at each position. The RPN undergoes end-to-end training to generate high-quality region proposals, subsequently employed by Fast R-CNN for detection. The integration of RPN and Fast R-CNN into a unified network involves sharing their convolutional features, with the RPN guiding the unified network on where to focus attention.
## Requirements

* torchvision==0.16.1
* pycocotools==2.0.7
* torch==2.1.1
* numpy==1.23.5
* opencv-python==4.8.1.78
* pillow==9.2.0
* An account in the [Dataloop platform](https://console.dataloop.ai/)

  
## Installation

To install the package and create the Faster R-CNN adapter, you will need a [project](https://developers.dataloop.ai/tutorials/getting_started/sdk_overview/chapter/#to-create-a-new-project) and a [dataset](https://developers.dataloop.ai/tutorials/data_management/manage_datasets/chapter/#create-dataset) in the Dataloop platform. The dataset should have [directories](https://developers.dataloop.ai/tutorials/data_management/manage_datasets/chapter/#create-directory) containing its training and validation subsets.

### Installing in the Platform

In the model management page, choose the AI Library button in the menu and in the drop-down menu, pick "Public Models" to see all the publicly available models. You will see fasterrcnn in the list and you can create a copy of it by selecting the "create model".

You will be presented with the options to choose name, artifact location and tags.

Then to choose between fine tuning or just choosing the pretrained weights. If you choose the pretrained weights, the model will be created with status ```trained```, otherwise, when choosing fine-tuning, you have to select a dataset, define the DQL filter or folder for the training and validation subsets, and choose a recipe for training. The model will be created with status ```created``` and you will need to run the training for it before it can be used.

Lastly, define the model configurations.


After this, the model will appear in the list of the proejct models in Model Management with the name you chose. It can be trained, evaluated and deployed.

### Installing via the SDK

To install Faster RCNN via SDK, all that is necessary is to clone the model from the AI Library to your own project:

```python
import dtlpy as dl
project = dl.projects.get('Faster RCNN Project')
public_model = dl.models.get(model_name="fasterrcnn")
model = project.models.clone(from_model=public_model,
                             model_name='fasterrcnn-clone',
                             project_id=project.id)
```

For more options when installing the model, check this [page](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#finetune-on-a-custom-dataset).

## Training and Fine-tuning

Training Faster RCNN segmentation can either be done via the platform or the SDK. For either purpose, it is necessary to first set the models subsets for training and validation. In the previous step, you saw how to define the train and validation subsets when creating your copy of the model. If you wish to do this via the SDK or modify them, you can follow [these instructions](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#define-dataset-subsets).

**ATTENTION:** To ensure that training will be successful, verify that the items in the dataset are annotated with annotations of type **polygon**. 

### Editing the configuration

To edit configurations via the platform, go to the Faster RCNN page in the Model Management and edit the json file displayed there or, via the SDK, by editing the model configuration. Click [here](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#model-configuration) for more information.

The basic configurations included are:

* ```epochs```: number of epochs to train the model (default: 50)
* ```batch_size```: batch size to be used during the training (default: 2)
* ```device```: whether to train on ```cpu``` or ```cuda``` (default to automatic detection of whether the instance has a GPU)
* ```labels```: The labels over which the model will train and predict (defaults to the labels in the model's dataset's recipe)
* ```id_to_label_map```: Dictionary mapping numbers to labels to guide the model outputs
* ```label_to_id_map```: Inverse map from ```id_to_label_map```

Additional configurations shown in the [Ultralytics documentation](https://docs.ultralytics.com/usage/cfg/#train) can be included in a dictionary under the key ```yaml_config```.

### Training with the Platform

In the Model Management page of your project, find a version of your Faster RCNN model with the status **created** and click the three dots in the right of the model's row and select the "Train" option.


Edit the configuration for this specific run of the training, and choose which instance in which it will run and select the service fields (more information [here](https://developers.dataloop.ai/tutorials/faas/custom_environment_using_docker/chapter/)):

Now kick back and wait for the training to finish.

### Training with the SDK

To train the model with the SDK, get the model id and define the service configuration for its training:

```python
model_entity = dl.models.get(model_id='<fasterrcnn-model-id>')
ex = model_entity.train()
ex.logs(follow=True)  # to stream the logs during training
custom_model = dl.models.get(model_id=model_entity.id)
print(custom_model.status)
```

For more information on how to customize the service configuration that will run the training, check the [documentation](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#train).

## Deployment

After installing the pretrained model or fine-tuning it on your data, it is necessary to deploy it, so it can be used for prediction.

### Deploying with the Platform

In the Model Management page of your project, find a pretrained or fine-tuned version of your Faster RCNN model and click the three dots in the right of the model's row and select the "Deploy" option:

Here you can choose the instance, minimum and maximum number of replicas and queue size of the service that will run the deployed model (for more information on these parameters, check [the documentation](https://developers.dataloop.ai/tutorials/faas/advance/chapter/#autoscaler)):


Proceed to the next page and define the service fields (which are explained [here](https://developers.dataloop.ai/tutorials/faas/custom_environment_using_docker/chapter/)).

After this, your model is deployed and ready to run inference.

### Deploying with the SDK

To deploy with the default service configuration defined in the package:

```python
model_entity = dl.models.get(model_id='<model-id>')
model_entity.deploy()
```

For more information and how to set specific service settings for the deployed model, check the [documentation](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#clone-and-deploy-a-model).

## Testing

Once the model is deployed, you can test it by going to the Model Management, selecting the Faster-RCNN model and then going to the test tab. Drag and drop or select an image to the image area, click the test button and wait for the prediction to be done/

## Prediction

### Predicting in the Platform

The best way to perform predictions in the platform is to add a "Predict Node" to a pipeline:

Click [here](https://developers.dataloop.ai/onboarding/08_pipelines/) for more information on Dataloop Pipelines.

### Predicting with the SDK

The deployed model can be used to run prediction on batches of images:

```python
model_entity = dl.models.get(model_id='<model-id>')
results = model_entity.predict_items([item_id_0, item_id_1, ..., item_id_n])
print(results)
```

For more information and options, [check the documentation](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#predict-items).

## Sources and Further Reading

* [PyTorch documentation](https://pytorch.org/vision/master/models/faster_rcnn.html)
