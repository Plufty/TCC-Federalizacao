from cgitb import handler
import os
import sys
import argparse
from time import time
import torch
from torch import nn
sys.path.append("../../")
from fedlab.core.network import DistNetwork
from fedlab.core.server.handler import AsyncParameterServerHandler
from fedlab.core.server.manager import AsynchronousServerManager
from fedlab.utils.functional import evaluate
from torchvision import datasets, models, transforms

import torchvision
import torchvision.transforms as transforms

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = True

def conf_matrix(fx, y, nome):

        if(nome == 'treino'):

          preds = fx.max(1, keepdim=True)[1]
          correct = y

          c = correct.tolist()
          p = preds.flatten().tolist()

          train_correct_list.append(c)
          train_predict_list.append(p)

          return train_correct_list, train_predict_list

        if(nome == 'validacao'):

          preds = fx.max(1, keepdim=True)[1]
          correct = y

          c = correct.tolist()
          p = preds.flatten().tolist()

          valid_correct_list.append(c)
          valid_predict_list.append(p)

          return valid_correct_list, valid_predict_list

        if(nome == 'teste'):

          preds = fx.max(1, keepdim=True)[1]
          correct = y

          c = correct.tolist()
          p = preds.flatten().tolist()

          test_correct_list.append(c)
          test_predict_list.append(p)

          return test_correct_list, test_predict_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distbelief training example')

    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=str, default='3002')
    parser.add_argument('--world_size', type=int)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int)

    args = parser.parse_args()

    #model = MLP()

    ###AlexNet###
    feature_extract = True
    model_ft = models.alexnet(pretrained=True)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs, 4)
    input_size = 224

    """ 
    #EfficientNet 
    feature_extract = True
    model_ft = EfficientNet.from_pretrained('efficientnet-b0')
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft._fc.in_features
    model_ft.classifier = nn.Linear(num_ftrs, num_classes)
    input_size = 224
    """
    
    
    """
    ###Resnet50###
    feature_extract = True
    model_ft = models.resnet50(pretrained=True)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 4)
    input_size = 224
    """


    """
    #ShuffleNet
    model_ft = models.shufflenet_v2_x1_0(pretrained=True)
    set_parameter_requires_grad(model_ft, True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 4)
    input_size = 224
    """

    """
    ###SqueezeNet###
    model_ft = models.squeezenet1_0(pretrained=True)
    set_parameter_requires_grad(model_ft, True)
    model_ft.classifier[1] = nn.Conv2d(512, 4, kernel_size=(1,1), stride=(1,1))
    model_ft.num_classes = 4
    input_size = 224
    """

        
    """
    ###VGG16###
    use_pretrained = True
    feature_extract = True
    model_ft = models.vgg16(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs, 4)
    input_size = 224
    """
    

    handler = AsyncParameterServerHandler(model_ft, alpha=0.5, total_time=5)

    network = DistNetwork(address=(args.ip, args.port),
                          world_size=args.world_size,
                          rank=0)

    t = time()

    Manager = AsynchronousServerManager(handler=handler, network=network)

    Manager.run()

    tend = time()




    batch_size = args.batch_size
    num_workers = 4

    data_dir = '../../dataset/'
    classes = ['Benign', 'Malignant']

    train_transforms = transforms.Compose([
        transforms.Resize(size=[224, 224]),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(size=[224, 224]),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    trains_dir = []
    valids_dir = []
    train_loaders = []
    valid_loaders = []
    test_loaders = []

    folds = os.listdir(data_dir)

    all_size_train = []
    all_size_valid = []

    for i in folds:
        test_dir = os.path.join(data_dir, i + '/test/')

        test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
        testloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
        test_loaders.append(testloader)

        print("\n\n\n Server Perspective")
        print('Num test images: ', len(test_data))


    criterion = nn.CrossEntropyLoss()

    print("Final Score Server: "+str(evaluate(model_ft, criterion, testloader)))
    print((tend-t)/60)
