import torch
import torch.nn as nn
import torch.nn.functional as F

class OurNet(nn.Module):
    def __init__(self, num_channels, gru_hidden_size, frame_height=224, frame_width=224):
        super(OurNet, self).__init__()

        # Input is the pre-convolved frames (potentially with optic flow channels)
        self.num_channels = num_channels
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.gru_hidden_size = gru_hidden_size

        # Flatten all channels per frame:
        # [batch, frames, C, H, W] -> [batch, frames, C * H * W]
        self.frame_flatten = nn.Flatten(start_dim=2)
        num_features = num_channels * frame_height * frame_width

        # TODO: Fully or semi-connected layer that loops GRU output as part of the next seq input.

        # Use out-of-the-box GRU for now.
        self.gru = nn.GRU(input_size=num_features,
                          hidden_size=gru_hidden_size, num_layers=1, bias=True,
                          batch_first=True, dropout=0, bidirectional=False)
        

        # ALTERNATE GRU IDEA:
        # Layer between the convolutional layer and GRU, which takes the
        # convolutional features PLUS the GRU output features as input,
        # and returns the same size. The purpose is to "learn" the importance of
        # features from the CNN and the GRU.
        # Consider this a pre-processing of the GRU input.
        # self.fc_combine = nn.Linear(in_features=2048 + output_size, out_features=2048 + output_size, bias=True)

        # GRU cell that will be looped over. We need to loop over a cell manually
        # (rather than a nn.GRU module that takes a whole sequence) so that we
        # can concatenate the output from each GRU cell to the input of the next.
        # input_size: output of a convolution of a frame (2048), plus the output.
        # self.gru = nn.GRUCell(input_size=2048 + output_size, hidden_size=gru_hidden_size, bias=True)


    def forward(self, input):
        # input.shape: [batch, frames, C, H, W]
        input = self.frame_flatten(input)

        # input.shape: [batch, frames, C * H * W]
        # gru_output.shape: [batch, frames, hidden_size]
        # h_n.shape: [1, batch, hidden_size]
        gru_output, h_n = self.gru(input)
        
        return gru_output

        # TODO: code for the custom GRU cell
        '''
        output = torch.tensor([])
        output = output.to(input.get_device())
        h_0 = self.initHidden()
        last_gru_out = self.initHidden()

        for frame in range(frames):
            # Pre-trained CNN expects one frame at a time
            cnn_out = self.cnn(input[:, f, :, :, :]).unsqueeze(0)
            print('cnn_out: ', cnn_out.size)
        '''

    # def initHidden(self):
    #     return torch.zeros(1, self.gru_hidden_size)


if __name__ == '__main__':
    from torchinfo import summary
    batch = 2
    frames = 30
    channels = 90
    height = 224
    width = 224

    input = torch.randn(batch, frames, channels, height, width)
    net = OurNet(num_channels=channels, gru_hidden_size=3,
                 frame_height=height, frame_width=width)

    output = net(input)

    print("input shape, [BATCH, FRAMES, C, H, W]: {}".format(input.shape))
    print("output shape, [BATCH, FRAMES, PREDICTION]: {}".format(output.shape))
    print()
    print(summary(net, input_size=(batch, frames, channels, height, width)))