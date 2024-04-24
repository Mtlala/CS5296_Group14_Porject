from model.vgg import VGG
from model.repvgg import RepVGG
from model.resnet import ResNet

def get_model(model:str):
    
    if model.startswith('VGG'):
        return VGG(model)
    
    elif model == "REPVGG":
        return RepVGG()
    
    elif model == "RESNET":
        return ResNet()
    
    else:
        return VGG('VGG11')