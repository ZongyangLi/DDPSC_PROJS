'''
Created on Jun 8, 2021

'''
import numpy as np
import os, sys, random, glob, shutil, argparse
import torch
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

# Class labels for prediction
class_names=['CannabisPlants','DryFlower','Objects']

# Preprocessing transformations
preprocess=transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

# Enable gpu mode, if cuda available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def options():
    
    parser = argparse.ArgumentParser(description='General Classifier(Cannabis)',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("-m", "--model_path", default='model.pth', help="model path")
    parser.add_argument("-i", "--in_dir", help="input directory")
    parser.add_argument("-o", "--out_dir", help="output directory")

    args = parser.parse_args()

    return args
         

# Perform prediction
def predict_class(image_path, model):
    
    with torch.no_grad():
        img=Image.open(image_path).convert('RGB')
        inputs=preprocess(img).unsqueeze(0).to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        label=class_names[preds]
    
    return label

def analisys_errors(in_dir, out_dir, model):
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
        
    image_file_suffix = '.jpg'
    # loop all images
    list_dirs = os.listdir(in_dir)
    for d in list_dirs:
        in_path = os.path.join(in_dir, d)
        if not os.path.isdir(in_path):
            continue
        
        class_name = d
        file_path_list = glob.glob(os.path.join(in_path, '*{}'.format(image_file_suffix)))
        for image_path in file_path_list:
            # predict
            pred_class = predict_class(image_path, model)
            # check if correct
            if pred_class != class_name:
                # save errors
                base_file_name = os.path.basename(image_path)
                dst_file_path = os.path.join(out_dir, '(B){}_(P){}_{}'.format(class_name, pred_class, base_file_name))
                shutil.copyfile(image_path, dst_file_path)
    
    return

def predict_and_save_to_different_dirs(in_dir, out_dir, model):
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    # init dst sub_folders
    for class_name in class_names:
        dst_folder = os.path.join(out_dir, class_name)
        if not os.path.isdir(dst_folder):
            os.mkdir(dst_folder)
        
    image_file_suffix = '.jpg'
    file_path_list = glob.glob(os.path.join(in_dir, '*{}'.format(image_file_suffix)))
    for image_path in file_path_list:
        # predict
        pred_class = predict_class(image_path, model)
        # save to dst dir
        base_file_name = os.path.basename(image_path)
        dst_file_path = os.path.join(out_dir, pred_class, base_file_name)
        shutil.copyfile(image_path, dst_file_path)
    
    return

def load_model(model_path):
    
    # Load the model for testing
    model = torch.load(model_path)
    model.eval()
    
    return model

def error_analysis_main():
    
    in_dir = '/media/zli/Elements/CannabisValidation/'
    out_dir = '/media/zli/Elements/CannabisErrorAnalysis/'
    model_path = '/home/zli/WorkSpace/PyWork/pytorch-py3/pytorch-image-classification-master/model.pth'
    
    model = load_model(model_path)
    
    analisys_errors(in_dir, out_dir, model)
    return

def main():
    
    args = options()
    model = load_model(args.model_path)
    predict_and_save_to_different_dirs(args.in_dir, args.out_dir, model)
    
    return


if __name__ == '__main__':
    
    main()