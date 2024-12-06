import timm

def ResNet18(num_classes:int=1000, in_chans=3, pretrained:bool=False, **kwargs):    
    model = timm.create_model('resnet18', pretrained=pretrained, num_classes=num_classes, **kwargs)
    return model

def ResNet34(num_classes:int=1000, in_chans=3, pretrained:bool=False, **kwargs): 
    model = timm.create_model('resnet34', pretrained=pretrained, num_classes=num_classes, **kwargs)
    return model

def ResNet50(num_classes:int=1000, in_chans=3, pretrained:bool=False, **kwargs):    
    model = timm.create_model('resnet50', pretrained=pretrained, num_classes=num_classes, **kwargs)
    return model

def ResNet101(num_classes:int=1000, in_chans=3, pretrained:bool=False, **kwargs):
    model = timm.create_model('resnet101', pretrained=pretrained, num_classes=num_classes, **kwargs)
    return model

def ResNet152(num_classes:int=1000, in_chans=3, pretrained:bool=False, **kwargs):  
    model = timm.create_model('resnet152', pretrained=pretrained, num_classes=num_classes, **kwargs)
    return model