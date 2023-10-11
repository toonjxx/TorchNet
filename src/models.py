from torch import nn

class NetInit(nn.modules):
    def weights_init(self):
        for m in self.modules():
            classname = m.__class__.__name__

            if "Conv" in classname:
                m.weight.data.normal_(0.0, 0.02)

            elif "BatchNorm" in classname:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        return x
    

    