import yaml
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

def register_coco_datasets(config: str) -> dict:
    """
    This function registers COCO datasets with detectron2.

    Args:
        config (str): The path to the configuration file.

    Returns:
        dict: A dictionary where the keys are the dataset names and the values are the corresponding MetadataCatalog objects.
    """
    with open(config, 'r') as file:
        config_dict = yaml.safe_load(file)
    output = {}
    for dataset_name, dataset_info in config_dict.items():
        register_coco_instances(
            name=dataset_name,
            metadata={},
            json_file=dataset_info['json_file'],
            image_root=dataset_info['root_dir']
        )
        DatasetCatalog.get(dataset_name)
        output[dataset_name] = MetadataCatalog.get(dataset_name)
    return output

def validate_labels(train_dataset: str, other_dataset: str) -> bool:
    """
    Validate thing_classes for each dataset.

    Args:
        train_dataset (str): The name of the training dataset.
        other_dataset (str): The name of the other dataset.

    Returns:
        bool: True if the labels for each item in the category ids is the same for both datasets, False otherwise.
    """
    train_metadata = MetadataCatalog.get(train_dataset)
    other_metadata = MetadataCatalog.get(other_dataset)
    
    if train_metadata.thing_classes != other_metadata.thing_classes:
        for train_label, other_label in zip(train_metadata.thing_classes, other_metadata.thing_classes):
            if train_label != other_label:
                raise ValueError(f"Labels do not match: {train_label} in train_dataset, {other_label} in other_dataset")
    return True

