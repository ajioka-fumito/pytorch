import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv_Conv(nn.Module):
    # conv→正規化→to_Reluを２回繰り返す。
    # 論文では出力と入力が一致していない
    # 解決案として縮まる画素数を逆算してpadiingをさしておいた
    def __init__(self,input_ch,output_ch):
        super(Conv_Conv,self).__init__()

        self.conv1 = nn.Conv2d(input_ch,output_ch,(3,3),stride=1,padding=1)
        self.conv2 = nn.Conv2d(output_ch,output_ch,(3,3),stride=1,padding=1)
        self.batchnorm = nn.BatchNorm2d(output_ch)

    def forward(self,x):
        # input
        x_input = x
        # first convolution
        conv1 = self.conv1(x)
        conv1 = self.batchnorm(conv1)
        conv1 = nn.Relu(conv1,inplace=True)
        # second convokution
        conv2 = self.conv2(conv1)
        conv2 = self.batchnorm(conv2)
        conv2 = nn.Relu(conv2,inplace=True)
        return conv2

class Down(nn.Module):
    # poolingで特徴抽出を行う。
    # 論文通りkernel_size=2,stride=2で設定
    # pytorchではデフォルトでstride=kernel_sizeとなるらしい
    def __init__(self,input_ch,putput_ch):
        super(Down,self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
    
    def forward(self,x):
        return self.pool(x)

class Up(nn.Module):
    # upsampling手法とconvtranspose手法が存在していて
    # 色々議論されている。
    # transposeを理解できなかった。
    # とりあえずupsampingで考える。

    def __init__(self):
        super(Up,self).__init__()

        self.upsampling = nn.Upsample(sacle_factor=2,mode="bilinear",align_corners=True)

    def forward(self,x):
        return self.upsampling(x)

