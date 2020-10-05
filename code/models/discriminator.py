import torch
import torch.nn as nn
import os

def make_discriminator(args, device, checkpoint):
    critic = Discriminator(in_channels=args.n_colors, 
                          out_features=1, 
                          blocks=args.d_blocks, 
                          features=args.d_features, 
                          bn=args.d_bn).to(device)
    
    if args.precision == 'half':
        critic.half()
        
    if args.resume:
        load_from = os.path.join(checkpoint.dir, 'model', 'critic_latest.pt')
        critic.load_state_dict_critic(torch.load(load_from))
    
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