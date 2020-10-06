import torch
import torchvision.models as models
import torch.nn as nn
import os

def make_discriminator(args, device, checkpoint, vgg=True):
    if not vgg:
        critic = Discriminator(in_channels=args.n_colors, 
                            out_features=1, 
                            blocks=args.d_blocks, 
                            features=args.d_features, 
                            bn=args.d_bn).to(device)
    else:
        critic = models.vgg11(True, True).to(device)
        critic = modifiy_vgg_model(critic, args.n_colors, 1, True, device=device)
        
    if args.precision == 'half':
        critic.half()
        
    if args.resume:
        load_from = os.path.join(checkpoint.dir, 'model', 'critic_latest.pt')
        # critic.load_state_dict_critic(torch.load(load_from))
        critic.load_state_dict(torch.load(load_from))
    
    return critic
    

class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, bn=False, pool=False):
        super().__init__()
        layers = []
        
        layers.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1))
        if bn: layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        
        layers.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1))
        if bn: layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        
        if pool:
            layers.append(nn.AvgPool2d(3, 2, 1))

        self.body = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.body(x)
    

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, out_features=1, blocks=8, features=64, bn=False):
        super().__init__()
        
        head = []
        pool = False
        head.append(DiscriminatorBlock(in_channels, features))
        for i in range(1, blocks):
            pool = True if i % 2 else False
            head.append(DiscriminatorBlock(features, features, bn, pool))
        
        self.pool = nn.AdaptiveAvgPool2d((10,10))

        # Linear End of the model
        lin_ftrs  = features * 10 * 10
        tail = []
        tail.append(nn.Linear(lin_ftrs, 2048))
        tail.append(nn.ReLU())
        tail.append(nn.Linear(2048, 256))
        tail.append(nn.ReLU())
        tail.append(nn.Linear(256, out_features))

        self.head = nn.Sequential(*head)
        self.tail = nn.Sequential(*tail)

    def forward(self, x):
        x = self.pool(self.head(x))
        x = x.view(x.shape[0], -1) # Flattening X
        return self.tail(x)
    
    def load_state_dict_critic(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                own_state[name].copy_(param)
            else:
                print(name)    

    
def clip_module_weights(m, min=0.01, max=0.01):
    if getattr(m, 'weight', False) is not False:
        m.weight.clamp_(min, max)
    if getattr(m, 'bias', False) is not False:
        m.bias.clamp_(min, max)
        

def change_vgg_input_channels(model:torch.nn.Module, channels=1):
    with torch.no_grad():
        nw = model.features[0].weight.clone()
        nw = nw[:, :channels, :, :].clone()
        model.features[0].weight = torch.nn.Parameter(nw.clone())
        model.features[0].in_channels = channels
    return model


def change_vgg_output_features(model:torch.nn.Module, out_features=1):
    with torch.no_grad():
        lin = model.classifier[-1]
        in_features = lin.in_features
        bias        = lin.bias is not None
        new_lin = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        model.classifier[-1] = new_lin
    return model


def remove_vgg_dropout(model:torch.nn.Module):
    with torch.no_grad():
        Dropout = torch.nn.modules.dropout.Dropout
        layers = []
        for layer in model.classifier:
            if not isinstance(layer, Dropout):
                layers.append(layer)
        model.classifier = nn.Sequential(*layers)
    return model


def modifiy_vgg_model(model, in_channels=None, out_features=1, remove_dropout=True, device='cpu'):
    if in_channels is not None:
        model = change_vgg_input_channels(model, in_channels)
    model = change_vgg_output_features(model, out_features)
    if remove_dropout:
        model = remove_vgg_dropout(model)
    return model.to(device)