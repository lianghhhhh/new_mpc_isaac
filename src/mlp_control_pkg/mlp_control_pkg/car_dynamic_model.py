import torch.nn as nn

# class CarDynamicModel(nn.Module):
#     """State-space style dynamics model — MLP variant.
    
#     Optimized for L4CasADi with widened layers and LayerNorm 
#     to compensate for the lack of recurrent temporal memory.
#     """
#     def __init__(self, input_size=800, output_size=19):
#         super(CarDynamicModel, self).__init__()
        
#         self.net = nn.Sequential(
#             nn.Linear(input_size, 1024),
#             nn.LayerNorm(1024),
#             nn.SiLU(),
            
#             nn.Linear(1024, 512),
#             nn.LayerNorm(512),
#             nn.SiLU(),
            
#             nn.Linear(512, 256),
#             nn.LayerNorm(256),
#             nn.SiLU(),
            
#             # Final output mapping
#             nn.Linear(256, output_size),
#         )

#     def forward(self, x):
#         # Accept either the flattened (B, input_size) vector directly, or
#         # the (B, seq_length, feat_dim) window shape used by the training loop
#         if x.dim() == 3:
#             x = x.reshape(x.shape[0], -1)
#         return self.net(x)


class CarDynamicModel(nn.Module):
    def __init__(self, input_size=400, output_size=19):
        super(CarDynamicModel, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            
            nn.Linear(64, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            
            nn.Linear(256, output_size),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.reshape(x.shape[0], -1)
        return self.net(x)