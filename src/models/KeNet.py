from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .modules import *

class KeNet(nn.Module):

    def __init__(self, norm_layer=None, zero_init_residual=True, p=0.5):
        super(KeNet, self).__init__()

        self.zero_init_residual = zero_init_residual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.srm = SRMConv2d(1, 0)
        self.bn1 = norm_layer(30)

        self.A1 = BlockA(30, 30, norm_layer=norm_layer)
        self.A2 = BlockA(30, 30, norm_layer=norm_layer)
        self.AA = BlockA(30, 30, norm_layer=norm_layer)

        # self.B1 = BlockB(30, 30, norm_layer=norm_layer)
        # self.B2 = BlockB(30, 64, norm_layer=norm_layer)

        self.B3 = BlockB(30, 64, norm_layer=norm_layer)
        self.A3 = BlockA(64, 64, norm_layer=norm_layer)

        self.B4 = BlockB(64, 128, norm_layer=norm_layer)
        self.A4 = BlockA(128, 128, norm_layer=norm_layer)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.bnfc = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        # self.fcfusion = nn.Linear(128, 128) #4
        self.fc = nn.Linear(128 * 4 + 1, 2)
        self.dropout = nn.Dropout(p=p)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.xavier_uniform_(m.weight)
                # nn.init.constant_(m.bias, 0.2)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)

        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, (BlockA, BlockB)):
                    nn.init.constant_(m.bn2.weight, 0)

    def extract_feat(self, x):
        x = x.float()
        out = self.srm(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.A1(out)
        out = self.A2(out)
        out = self.AA(out)

        # out = self.B1(out)
        # out = self.B2(out)

        out = self.B3(out)
        out = self.A3(out)

        out = self.B4(out)
        out = self.A4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), out.size(1))

        # out = self.relu(out)
        # out = self.bnfc(out)

        return out

    def forward(self, *args):
        ############# statistics fusion start #############
        feats = torch.stack(
            [self.extract_feat(subarea) for subarea in args], dim=0
        )

        euclidean_distance = F.pairwise_distance(feats[0], feats[1], eps=1e-6,
                                                 keepdim=True)

        if feats.shape[0] == 1:
            final_feat = feats.squeeze(dim=0)
        else:
            # feats_sum = feats.sum(dim=0)
            # feats_sub = feats[0] - feats[1]
            feats_mean = feats.mean(dim=0)
            feats_var = feats.var(dim=0)
            feats_min, _ = feats.min(dim=0)
            feats_max, _ = feats.max(dim=0)

            '''feats_sum = feats.sum(dim=0)
            feats_sub = abs(feats[0] - feats[1])
            feats_prod = feats.prod(dim=0)
            feats_max, _ = feats.max(dim=0)'''
            
            #final_feat = torch.cat(
            #    [feats[0], feats[1], feats[0], feats[1]], dim=-1
            #    #[euclidean_distance, feats_sum, feats_sub, feats_prod, feats_max], dim=-1
            #)

            final_feat = torch.cat(
                [euclidean_distance, feats_mean, feats_var, feats_min, feats_max], dim=-1
                #[euclidean_distance, feats_sum, feats_sub, feats_prod, feats_max], dim=-1
            )

        out = self.dropout(final_feat)
        # out = self.fcfusion(out)
        # out = self.relu(out)
        out = self.fc(out)

        return out, feats[0], feats[1]
