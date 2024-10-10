#!/usr/bin/env python

import os
import json
#import argparse
import fiftyone as fo
import fiftyone.zoo as foz

#parser = argparse.ArgumentParser(description='Download OpenImages using FiftyOne', add_help=True)
# parser.add_argument('--dataset-dir', default='/open-images-v6', help='dataset download location')
dataset_dir = '/open-images-v6'
 
#parser.add_argument('--splits', default=['train', 'validation'], choices=['train', 'validation', 'test'],
 #                   nargs='+', type=str,
  #                  help='Splits to download, possible values are train, validation and test')
splits = ['train', 'validation']
#parser.add_argument('--classes', default=None, nargs='+', type=str,
 #                   help='Classes to download. default to all classes')
classes = None

#parser.add_argument('--output-labels', default='labels.json', type=str,
 #                   help='Classes to download. default to all classes')

output_labels = 'labels.json'
#args = parser.parse_args()
max_samples=300

print("Downloading open-images dataset ...")
dataset = foz.load_zoo_dataset(
    name="open-images-v6",
    classes=classes,
    splits=splits,
    label_types="detections",
    dataset_name="open-images",
    dataset_dir=dataset_dir,
    max_samples=max_samples
)

print("Converting dataset to coco format ...")
for split in splits:
    output_fname = os.path.join(dataset_dir, split, "labels", output_labels)
    split_view = dataset.match_tags(split)
    split_view.export(
        labels_path=output_fname,
        dataset_type=fo.types.COCODetectionDataset,
        label_field="detections",
        classes=classes)

    # Add iscrowd label to openimages annotations
    with open(output_fname) as fp:
        labels = json.load(fp)
    for annotation in labels['annotations']:
        annotation['iscrowd'] = int(annotation['IsGroupOf'])
    with open(output_fname, "w") as fp:
        json.dump(labels, fp)
