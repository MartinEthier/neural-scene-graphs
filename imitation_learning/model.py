import torch
from torch import nn
import torch.nn.functional as F
import timm


class E2EModel(nn.Module):
    """
    Initializes a pretrained ImageNet feature extractor from timm.
    Input images have to be at least 224x224, scaled to be within 0 and 1, then normalized using:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    """
    def __init__(self, cfg):
        super(E2EModel, self).__init__()
        self.cfg = cfg
        
        # Initialize pretrained model with 0 classes to get pooled features
        self.timm_encoder = timm.create_model(
            cfg['name'],
            pretrained=True,
            num_classes=0,
            in_chans=3*cfg['num_frames'],
            drop_rate=cfg['dropout_prob']
        )
        
        # Add extra fc layers and map to desired output size
        self.fc = nn.Linear(in_features=cfg['timm_feat_size'], out_features=cfg['fc_size'], bias=False)
        self.bn = nn.BatchNorm1d(num_features=cfg['fc_size'])
        self.dropout = nn.Dropout(cfg['dropout_prob'])
        self.fc_out = nn.Linear(in_features=cfg['fc_size'], out_features=cfg['output_size'])

    def forward(self, X):
        X = self.timm_encoder(X) # (batch_size, feature_len)
        X = self.fc(X) # (batch_size, out_features)
        X = self.bn(X)
        X = F.leaky_relu(X)
        X = self.dropout(X)
        X = self.fc_out(X)

        # Reshape from (B, horizon*2) to (B, horizon, 2)
        X = X.view(X.shape[0], -1, 2)

        return X

if __name__=="__main__":
    cfg = {
        "name": "resnet34",
        "timm_feat_size": 512,
        "fc_size": 256,
        "output_size": 50*2,
        "dropout_prob": 0.2,
        "num_frames": 2

    }
    model = E2EModel(cfg) 
    output = model(torch.randn(4, 6, 256, 256))
    print(output.shape)