"""
        This function creates an CRF as RNN model. See this https://www.robots.ox.ac.uk/~szheng/papers/CRFasRNN.pdf for
        details.
        It depends on https://github.com/sadeepj/crfasrnn_keras/ and https://github.com/MiguelMonteiro/CRFasRNNLayer
"""
import torch.nn as nn
from Network.Blocks.Permutohedral_Filtering import PermutohedralLayer
import configparser
import torch

class CRFasRNN(nn.Module):

    def __init__(
        self,
        network,
        nb_iterations,
        nb_classes,
        nb_input_channels,
        theta_alpha,
        theta_beta,
        theta_gamma,
        gpu_rnn,
        train_mode
    ):
        """

        The crf as rnn model combines a neural network with a crf model. The crf can be written as recurrent neural net
        work. The network delivers the unaries. The rnn uses mean field approximation to improve the results
        :param network: The network to deliver the unaries. The input must have the shape [batch, nb_input_channels,
        width, height]. The output must have the shape [batch, nb_classes, width, height]
        :param nb_iterations: Who many iterations does the rnn run. In the paper 5 are used for train and 10 for test
        :param nb_classes: Who many segments does the output have
        :param nb_input_channels: number of color channels. should be 1 or 3.
        :param theta_alpha: used for permutohedral filter. See paper for details
        :param theta_beta:used for permutohedral filter. See paper for details
        :param theta_gamma:used for permutohedral filter. See paper for details
        :param gpu_rnn: On which GPU does the RNN run. None if CPU.
        """
        super(CRFasRNN, self).__init__()
        self.use_gpu = gpu_rnn is not None
        self.gpu = gpu_rnn
        self.network = network
        self.nb_iterations = nb_iterations
        self.nb_classes = nb_classes
        self.nb_input_channels = nb_input_channels

        # check whether cnn should be updated
        if train_mode.lower() == "crf":
            for param in self.network.parameters():
                param.requires_grad = False

        # These are the elements for the filtering
        self.theta_alpha = theta_alpha
        self.theta_beta = theta_beta
        self.theta_gamma = theta_gamma



        self.spatial_filter = PermutohedralLayer(
            bilateral=False,
            theta_alpha=self.theta_alpha,
            theta_beta=self.theta_beta,
            theta_gamma=self.theta_gamma
        )

        # This is the convolutional layer for spatial filtering
        self.spatial_conv = nn.Conv2d(
            in_channels=nb_classes,
            out_channels=nb_classes,
            kernel_size=(1, 1)
        )

        self.bilateral_filter = PermutohedralLayer(
            bilateral=True,
            theta_alpha=self.theta_alpha,
            theta_beta=self.theta_beta,
            theta_gamma=self.theta_gamma
        )

        # This is the convolutional layer for bilateral filtering
        self.bilateral_conv = nn.Conv2d(
            in_channels=nb_classes,
            out_channels=nb_classes,
            kernel_size=(1, 1)
        )

        # This is the convolutional layer for compatiblity step
        self.compatitblity_conv = nn.Conv2d(
            in_channels=nb_classes,
            out_channels=nb_classes,
            kernel_size=(1, 1)
        )



        # push all the thing to the gpu
        if self.use_gpu:
            self.bilateral_conv.cuda(gpu_rnn)
            self.compatitblity_conv.cuda(gpu_rnn)
            self.bilateral_filter.cuda(gpu_rnn)
            self.spatial_conv.cuda(gpu_rnn)
            self.spatial_filter.cuda(gpu_rnn)

        # check whether crf should be updated
        if train_mode.lower() == "cnn":
            self.bilateral_conv.weight.requires_grad = False
            self.bilateral_conv.bias.requires_grad = False
            self.compatitblity_conv.weight.requires_grad = False
            self.compatitblity_conv.bias.requires_grad = False
            self.spatial_conv.weight.requires_grad = False
            self.spatial_conv.bias.requires_grad = False

        # softmax
        self.softmax = nn.Softmax2d()

    def forward(
            self,
            image,
            name=None
    ):

        # calculate the unaries
        image = image.cuda(1)
        unaries = self.network(image)
        if self.use_gpu:
            unaries = unaries.cuda(self.gpu)

        # set the q_values
        image = image.cuda(0)
        q_values = unaries
        softmax_out = self.softmax(q_values)
        for i in range(self.nb_iterations):

            # 1. Filtering
            # 1.1 spatial filtering
            spatial_out = self.spatial_filter(
                softmax_out,
                image
            )
            # 1.2 bilateral filtering
            bilateral_out = self.bilateral_filter(
                softmax_out,
                image
            )
            # 2. weighted filter outputs
            message_passing = self.spatial_conv(spatial_out) + self.bilateral_conv(bilateral_out)

            # 3. compatibilty transform
            pairwise = self.compatitblity_conv(message_passing)

            # 4. add pairwise terms
            q_values = unaries - pairwise

            # 5. Softmax
            softmax_out = self.softmax(q_values)

        return softmax_out

    def crf_dict(self):
        return {
            "spatial_conv": self.spatial_conv.state_dict(),
            "bilateral_conv": self.bilateral_conv.state_dict(),
            "compatitblity_conv": self.compatitblity_conv.state_dict()
        }

    def cnn_dict(self):
        return self.network.state_dict()

    def cnn_parameters(self):
        return self.network.parameters()

    def crf_parameters(self):
        return [
            self.spatial_conv.bias,
            self.spatial_conv.weight,
            self.bilateral_conv.bias,
            self.bilateral_conv.weight,
            self.compatitblity_conv.bias,
            self.compatitblity_conv.weight
        ]

    def load_parameter(
            self,
            cnn_path=None,
            crf_path=None
    ):
        if cnn_path is not None:
            self.network.load_state_dict(torch.load(cnn_path))
        if crf_path is not None:
            state_dict = torch.load(crf_path)
            self.spatial_conv.load_state_dict(state_dict["spatial_conv"])
            self.bilateral_conv.load_state_dict(state_dict["bilateral_conv"])
            self.compatitblity_conv.load_state_dict(state_dict["compatitblity_conv"])
