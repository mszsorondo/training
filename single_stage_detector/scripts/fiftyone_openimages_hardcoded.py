#!/usr/bin/env python

import os
import json
#import argparse
#if error mongodb: 
#os.environ["FIFTYONE_DATABASE_VALIDATION"] = "0"
import fiftyone as fo
import fiftyone.zoo as foz
import pdb
#parser = argparse.ArgumentParser(description='Download OpenImages using FiftyOne', add_help=True)
# parser.add_argument('--dataset-dir', default='/open-images-v6', help='dataset download location')

dataset_dir = r'C:/Users/Usuario-PC/fiftyone/open-images-v6'

#parser.add_argument('--splits', default=['train', 'validation'], choices=['train', 'validation', 'test'],
 #                   nargs='+', type=str,
  #                  help='Splits to download, possible values are train, validation and test')
splits = ['train', 'validation']
#parser.add_argument('--classes', default=None, nargs='+', type=str,
 #                   help='Classes to download. default to all classes')
classes = None

#parser.add_argument('--output-labels', default='labels.json', type=str,
 #                   help='Classes to download. default to all classes')

output_labels = 'openimages-mlperf.json'
#args = parser.parse_args()
max_samples=300

print("Downloading open-images dataset ...")
dataset = foz.load_zoo_dataset(
    name_or_url="open-images-v6",
    #label_field="ground_truth",
    splits=splits,
    dataset_name="open-images",
    max_samples=max_samples
)
breakpoint()
print("Converting dataset to coco format ...")
for split in splits:
    output_fname = os.path.join(dataset_dir, split, "labels", output_labels)
    split_view = dataset.match_tags(split)
    #breakpoint()
    #split_view.export(labels_path=output_fname, dataset_type=fo.types.COCODetectionDataset,label_field="detections",classes=classes)
    split_view.export(
        labels_path=output_fname,
        dataset_type=fo.types.COCODetectionDataset,
        label_field="detections", #si tira error
        #label_field='detections_detections',
        classes=classes)
    # Add iscrowd label to openimages annotations
    with open(output_fname) as fp:
        labels = json.load(fp)
    for annotation in labels['annotations']:
        annotation['iscrowd'] = int(annotation['IsGroupOf'])
    with open(output_fname, "w") as fp:
        json.dump(labels, fp)