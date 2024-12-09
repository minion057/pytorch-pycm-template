import timm

def res2net50d(num_classes:int=1000, in_chans=3, pretrained:bool=False, **kwargs):    
    model = timm.create_model('res2net50d', pretrained=pretrained, num_classes=num_classes, **kwargs)
    return model
def res2next50(num_classes:int=1000, in_chans=3, pretrained:bool=False, **kwargs):    
    model = timm.create_model('res2next50', pretrained=pretrained, num_classes=num_classes, **kwargs)
    return model


def inception_v3(num_classes:int=1000, in_chans=3, pretrained:bool=False, **kwargs):    
    model = timm.create_model('inception_v3', pretrained=pretrained, num_classes=num_classes, **kwargs)
    return model
def xception(num_classes:int=1000, in_chans=3, pretrained:bool=False, **kwargs):    
    model = timm.create_model('xception', pretrained=pretrained, num_classes=num_classes, **kwargs)
    return model
def inception_resnet_v2(num_classes:int=1000, in_chans=3, pretrained:bool=False, **kwargs):    
    model = timm.create_model('inception_resnet_v2', pretrained=pretrained, num_classes=num_classes, **kwargs)
    return model

def dense121(num_classes:int=1000, in_chans=3, pretrained:bool=False, **kwargs):    
    model = timm.create_model('densenet121.ra_in1k', pretrained=pretrained, num_classes=num_classes, **kwargs)
    return model