import os
import argparse
import logging
import cv2
import torch
import torchvision
import numpy as np
import dtlpy as dl

from PIL import Image
from glob import glob
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision import transforms as T
from dtlpy.utilities.dataset_generators.dataset_generator_torch import DatasetGeneratorTorch
from utils.engine import train_one_epoch, evaluate

logger = logging.getLogger('FasterRCNNAdapter')


@dl.Package.decorators.module(
    description='Model Adapter for FasterRCNN object detection',
    name='model-adapter',
    init_inputs={'model_entity': dl.Model}
    )
class FasterRCNNAdapter(dl.BaseModelAdapter):

    def __init__(self, model_entity: dl.Model = None):
        super().__init__(model_entity)

    def get_model_instance_segmentation(self, num_classes):
        # Reading configs:
        hidden_layer = self.configuration.get("hidden_layer", 256)

        # load an instance segmentation model pre-trained on COCO
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            num_classes,
            )
        return model

    @staticmethod
    def dl_collate(batch):
        ims = torch.stack([torch.transpose(b['image'].float(), 2, 1) for b in batch])
        tgs = list()
        for b in batch:
            masks = list()
            for seg in b['segment']:
                mask = np.zeros(shape=b['image'].shape[1:])
                mask = cv2.drawContours(
                    image=mask,
                    contours=[np.asarray(seg).astype('int')],
                    contourIdx=-1,
                    color=True,
                    thickness=-1
                    )
                masks.append(mask)
            boxes = torch.as_tensor(b['box'], dtype=torch.float32)
            masks = torch.as_tensor(np.asarray(masks), dtype=torch.uint8)
            labels = torch.Tensor(b['class']).to(torch.int64)
            tgs.append(
                {'boxes': boxes,
                 'area': [(box[3] - box[1]) * (box[2] - box[0]) for box in boxes],
                 "iscrowd": [True] * len(labels),
                 'labels': labels,
                 'masks': masks,
                 'image_id': b['item_id']}
                )

        return ims, tgs

    def load(self, local_path, **kwargs):
        num_classes = len(self.model_entity.labels)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model_filename = os.path.join(local_path, self.configuration.get('model_filename', 'weights/best.pt'))

        self.model = self.get_model_instance_segmentation(num_classes)
        os.makedirs(local_path, exist_ok=True)
        if os.path.exists(model_filename):
            logger.info("Loading saved weights")
            self.model.load_state_dict(torch.load(model_filename, map_location=device))
        else:
            logger.info("No weights file found. Loading pre-trained weights.")

    def save(self, local_path, **kwargs):
        model_filename = os.path.join(local_path, self.configuration.get('model_filename', 'weights/best.pt'))
        torch.save(self.model.state_dict(), model_filename)
        logger.info(f"Saved state dict at {model_filename}")
        self.configuration.update({'model_filename': 'weights/best.pt'})

    def prepare_item_func(self, item: dl.entities.Item):
        img_size = self.configuration.get("input_size", 256)
        buffer = item.download(save_locally=False)
        image = np.asarray(Image.open(buffer))
        if image.shape[0] != 3:
            image = np.transpose(image, (2, 0, 1))
            image = np.resize(image, (3, img_size, img_size))
        return image

    def predict(self, batch, **kwargs):
        # Reading configs:
        conf_threshold = self.configuration.get('conf_threshold', 0.5)
        id_to_label_map = self.configuration['id_to_label_map']

        self.model.eval()
        logger.info("Model set to evaluation mode")
        results = self.model(torch.Tensor(batch))
        logger.info("Batch prediction finished")
        batch_annotations = list()
        logger.info("Creating annotations based on predictions")
        for i_img, result in enumerate(results):
            logger.info(f"Annotations for item #{i_img}. Total number of detections: {len(result['labels'])}")
            image_annotations = dl.AnnotationCollection()
            for i_pred in range(len(result['labels'])):
                logger.info(f"Detection #{i_pred} for item #{i_img}")
                score = float(result['scores'][i_pred])
                if score < conf_threshold:
                    logger.info(
                        f"Ignoring detection, because its confidence ({score}) "
                        f"is lower than the threshold of {conf_threshold}"
                        )
                    continue
                cls = int(result['labels'][i_pred])
                mask = result['masks'][i_pred].cpu().detach().numpy().squeeze()
                logger.info(f"Class: {id_to_label_map[str(cls)]}, confidence: {score}")
                image_annotations.add(
                    annotation_definition=dl.Polygon.from_segmentation(
                        mask=mask,
                        label=id_to_label_map[str(cls)]
                        ),
                    model_info={'name': self.model_entity.name,
                                'model_id': self.model_entity.id,
                                'confidence': score}
                    )
            batch_annotations.append(image_annotations)
        logger.info("Annotations created successfully")
        return batch_annotations

    def train(self, data_path, output_path, **kwargs):
        # Reading configs:
        num_epochs = self.configuration.get("num_epochs", 10)
        input_size = self.configuration.get("input_size", 256)
        train_batch_size = self.configuration.get("train_batch_size", 12)
        val_batch_size = self.configuration.get("val_batch_size", 1)
        dataloader_num_workers = self.configuration.get("num_workers", 1)
        optim_learning_rate = self.configuration.get('learning_rate', 0.005)
        optim_momentum = self.configuration.get('momentum', 0.9)
        optim_weight_decay = self.configuration.get('weight_decay', 0.0005)
        scheduler_step_size = self.configuration.get('step_size', 3)
        scheduler_gamma = self.configuration.get('gamma', 0.1)
        id_to_label_map = self.model_entity.configuration.get("id_to_label_map")
        label_to_id_map = self.model_entity.configuration.get("label_to_id_map")
        train_filter = self.model_entity.metadata['system']['subsets']['train']['filter']
        val_filter = self.model_entity.metadata['system']['subsets']['validation']['filter']
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        logger.info(f"Using device: {device}")

        os.makedirs(output_path, exist_ok=True)
        os.makedirs(os.path.join(output_path, 'weights'), exist_ok=True)
        logger.info("Model set to train mode.")

        def get_transform(train=True):
            transforms = list()
            transforms.append(T.ToTensor())
            if train:
                transforms.append(T.RandomHorizontalFlip(0.5))
            transforms.append(T.Resize((input_size, input_size)))
            return T.Compose(transforms)

        logger.debug("Trainset generator created")
        train_dataset = DatasetGeneratorTorch(
            data_path=os.path.join(data_path, 'train'),
            filters=dl.Filters(custom_filter=train_filter),
            dataset_entity=self.model_entity.dataset,
            id_to_label_map=id_to_label_map,
            label_to_id_map=label_to_id_map,
            overwrite=False,
            to_mask=False,
            annotation_type=dl.AnnotationType.POLYGON,
            transforms=get_transform()
            )
        val_dataset = DatasetGeneratorTorch(
            data_path=os.path.join(data_path, 'validation'),
            filters=dl.Filters(custom_filter=val_filter),
            dataset_entity=self.model_entity.dataset,
            id_to_label_map=id_to_label_map,
            label_to_id_map=label_to_id_map,
            overwrite=False,
            to_mask=False,
            annotation_type=dl.AnnotationType.POLYGON,
            transforms=get_transform(False)
            )

        data_loader = torch.utils.data.DataLoader(
            train_dataset,
            train_batch_size,
            shuffle=True,
            num_workers=dataloader_num_workers,
            collate_fn=self.dl_collate
            )
        logger.debug("Train data loader created")
        data_loader_test = torch.utils.data.DataLoader(
            val_dataset,
            val_batch_size,
            shuffle=True,
            num_workers=dataloader_num_workers,
            collate_fn=self.dl_collate
            )
        logger.debug("Val data loader created")
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params,
            lr=optim_learning_rate,
            momentum=optim_momentum,
            weight_decay=optim_weight_decay
            )

        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_step_size,
            gamma=scheduler_gamma
            )
        logger.debug("Starting training")

        def epoch_end_callback(metrics, epoch):
            samples = list()
            for meter, value in metrics.meters.items():
                if 'loss' in meter:
                    legend = 'val' if 'val' in meter else 'train'
                    figure = meter.split('val_')[-1] if legend == 'val' else meter
                    samples.append(dl.PlotSample(figure=figure, legend=legend, x=epoch, y=value.value))
            self.model_entity.metrics.create(samples, dataset_id=self.model_entity.dataset_id)

        def eval_results_callback(metrics, epoch):
            samples = list()
            for iou_metric, coco_eval in metrics.coco_eval.items():
                stats = coco_eval.stats
                for i, stat in enumerate(stats):
                    metric_name = "AP" if i <= 5 else "AR"
                    if i == 1:
                        iou_thresh = "@IoU=0.5"
                    elif i == 2:
                        iou_thresh = "@IoU=0.75"
                    else:
                        iou_thresh = "@IoU=0.5:0.95"
                    if i in [3, 9]:
                        area = "|area=small"
                    elif i in [4, 10]:
                        area = "|area=medium"
                    elif i in [5, 11]:
                        area = "|area=large"
                    else:
                        area = "|area=all"
                    if i == 6:
                        max_dets = "|max_dets=1"
                    elif i == 7:
                        max_dets = "|max_dets=10"
                    else:
                        max_dets = "|max_dets=100"
                    figure = iou_metric + metric_name + iou_thresh + area + max_dets
                    samples.append(dl.PlotSample(figure=figure, legend="val", x=epoch, y=stat))
            self.model_entity.metrics.create(samples, dataset_id=self.model_entity.dataset_id)

        for epoch in range(num_epochs):
            logger.debug(f"Training epoch {epoch}")
            epoch_metrics = train_one_epoch(
                self.model,
                optimizer,
                data_loader,
                device=device,
                val_data_loader=data_loader_test,
                epoch=epoch,
                print_freq=10
                )
            epoch_end_callback(epoch_metrics, epoch)
            lr_scheduler.step()
            eval_results = evaluate(self.model, data_loader_test, device=device)
            eval_results_callback(eval_results, epoch)
        logger.info("Training finished successfully")

    def convert_from_dtlpy(self, data_path, **kwargs):
        input_size = self.configuration.get("input_size", 256)
        subsets = list(self.model_entity.metadata['system']['subsets'].keys())
        subpaths = [self.model_entity.metadata['system']['subsets'][subset]['filter']['$and'][0]['dir'][1:] \
                    for subset in subsets]

        for subset, subpath in zip(subsets, subpaths):
            img_paths = glob(os.path.join(data_path, subset, 'items', subpath, '*'))
            for img_path in img_paths:
                img = Image.open(img_path)
                img.resize((input_size, input_size))
                img.save(img_path)


def package_creation(project):
    metadata = dl.Package.get_ml_metadata(
        cls=FasterRCNNAdapter,
        default_configuration={'weights_filename': 'weights/best.pt',
                               'input_size': 256},
        output_type=dl.AnnotationType.CLASSIFICATION,
        )
    module = dl.PackageModule.from_entry_point(entry_point='model_adapter.py')
    package = project.packages.push(
        package_name='faster-rcnn',
        src_path=os.getcwd(),
        is_global=False,
        package_type='ml',
        modules=[module],
        service_config={
            'runtime': dl.KubernetesRuntime(
                pod_type=dl.INSTANCE_CATALOG_REGULAR_M,
                autoscaler=dl.KubernetesRabbitmqAutoscaler(
                    min_replicas=0,
                    max_replicas=1
                    ),
                concurrency=1
                ).to_json()},
        metadata=metadata
        )
    return package


def model_creation(package: dl.Package, project: dl.Project, dataset: dl.Dataset):
    model = package.models.create(
        model_name='faster-rcnn',
        description='faster-rcnn for object segmentation',
        tags=['pretrained'],
        dataset_id=dataset.id,
        scope='project',
        status='created',
        configuration={'weights_filename': 'weights/best.pth',
                       'batch_size': 12,
                       'num_epochs': 25,
                       'id_to_label_map': {i: x.tag for i, x in enumerate(dataset.labels)},
                       'label_to_id_map': {x.tag: i for i, x in enumerate(dataset.labels)}
                       },
        project_id=project.id,
        labels=[x.tag for x in dataset.labels],
        output_type='segment',
        input_type='image'
        )
    return model


def parse_args():
    arg_parser= argparse.ArgumentParser()
    arg_parser.add_argument("--env", "-e", type=str, help="Environment (prod or rc)")
    arg_parser.add_argument("--project", "-p", type=str, help="Project name")
    arg_parser.add_argument("--dataset", "-d", type=str, help="Dataset name")
    return arg_parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    env = args.env
    project_name = args.project
    dataset_name = args.dataset
    dl.setenv(env)
    project = dl.projects.get(project_name)
    package = package_creation(project)
    dataset = project.datasets.get(dataset_name)
    model = model_creation(package, project, dataset)
    print(
        f"Model {model.name} created with dataset {dataset.name}"
        f"with package {package.name} in project {project.name}!"
        )
