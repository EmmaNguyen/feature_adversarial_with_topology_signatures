import torch.nn as nn
import torch.nn.functional as F


class MNISTEncoder(nn.Module):
    """MNIST Encoder from Original Paper Keras based Implementation."""
    def __init__(self, init_num_filters=16, lrelu_slope=0.2, inter_fc_dim=128, embedding_dim=2):
        super(MNISTEncoder, self).__init__()

        self.init_num_filters_ = init_num_filters
        self.lrelu_slope_ = lrelu_slope
        self.inter_fc_dim_ = inter_fc_dim
        self.embedding_dim_ = embedding_dim

        self.features = nn.Sequential(
            nn.Conv2d(1, self.init_num_filters_ * 1, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),

            nn.Conv2d(self.init_num_filters_ * 1, self.init_num_filters_ * 1, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),

            nn.AvgPool2d(kernel_size=2, padding=0),
            nn.Conv2d(self.init_num_filters_ * 1, self.init_num_filters_ * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),

            nn.Conv2d(self.init_num_filters_ * 2, self.init_num_filters_ * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),

            nn.AvgPool2d(kernel_size=2, padding=0),
            nn.Conv2d(self.init_num_filters_ * 2, self.init_num_filters_ * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),

            nn.Conv2d(self.init_num_filters_ * 4, self.init_num_filters_ * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),

            nn.AvgPool2d(kernel_size=2, padding=1)
        )
        self.fc = nn.Sequential(
            nn.Linear(self.init_num_filters_ * 4 * 4 * 4, self.inter_fc_dim_),
            nn.ReLU(inplace=True),
            nn.Linear(self.inter_fc_dim_, self.embedding_dim_)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.init_num_filters_ * 4 * 4 * 4)
        x = self.fc(x)
        return x


class MNISTDecoder(nn.Module):
    """MNIST Decoder from Original Paper Implementation."""
    def __init__(self, init_num_filters=16, lrelu_slope=0.2, inter_fc_dim=128, embedding_dim=2):
        super(MNISTDecoder, self).__init__()

        self.init_num_filters_ = init_num_filters
        self.lrelu_slope_ = lrelu_slope
        self.inter_fc_dim_ = inter_fc_dim
        self.embedding_dim_ = embedding_dim

        self.fc = nn.Sequential(
            nn.Linear(self.embedding_dim_, self.inter_fc_dim_),
            nn.Linear(self.inter_fc_dim_, self.init_num_filters_ * 4 * 4 * 4),
            nn.ReLU(inplace=True)
        )

        self.features = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.init_num_filters_ * 4, self.init_num_filters_ * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),

            nn.Conv2d(self.init_num_filters_ * 4, self.init_num_filters_ * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.init_num_filters_ * 4, self.init_num_filters_ * 4, kernel_size=3, padding=0),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),

            nn.Conv2d(self.init_num_filters_ * 4, self.init_num_filters_ * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.init_num_filters_ * 4, self.init_num_filters_ * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),

            nn.Conv2d(self.init_num_filters_ * 2, self.init_num_filters_ * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),

            nn.Conv2d(self.init_num_filters_ * 2, 1, kernel_size=3, padding=1)
        )

    def forward(self, z):
        z = self.fc(z)
        z = z.view(-1, 4 * self.init_num_filters_, 4, 4)
        z = self.features(z)
        return F.sigmoid(z)

class MNISTDiscriminator(nn.Module):
    """MNIST Discriminator from [Citation]"""
    def __init__(self, subscripted_views):
        super(PHConvNet, self).__init__()
        self.subscripted_views = subscripted_views
        n_elements = 75
        n_filters = 32
        stage_2_out = 15
        n_neighbor_directions = 1
        output_size = 10
        self.transform = UpperDiagonalThresholdedLogTransform(0.1)

        # Stacking
        self.pht_sl = SLayerPHT(len(subscripted_views),n_elements,2,n_neighbor_directions=n_neighbor_directions,
                                center_init=self.transform(pers_dgm_center_init(n_elements)),
                                sharpness_init=torch.ones(n_elements, 2) * 4)
        self.stage_1 = []
        for i in range(len(subscripted_views)):
            seq = nn.Sequential()
            seq.add_module('conv_1', nn.Conv1d(1 + 2 * n_neighbor_directions, n_filters, 1, bias=False))
            seq.add_module('conv_2', nn.Conv1d(n_filters, 8, 1, bias=False))
            self.stage_1.append(seq)
            self.add_module('stage_1_{}'.format(i), seq)

        self.stage_2 = []
        for i in range(len(subscripted_views)):
            seq = nn.Sequential()
            seq.add_module('linear_1', nn.Linear(n_elements, stage_2_out))
            seq.add_module('batch_norm', nn.BatchNorm1d(stage_2_out))
            seq.add_module('linear_2', nn.Linear(stage_2_out, stage_2_out))
            seq.add_module('relu', nn.ReLU())
            seq.add_module('Dropout', nn.Dropout(0.4))
            self.stage_2.append(seq)
            self.add_module('stage_2_{}'.format(i), seq)

        linear_1 = nn.Sequential()
        linear_1.add_module('linear', nn.Linear(len(subscripted_views) * stage_2_out, 50))
        linear_1.add_module('batchnorm', torch.nn.BatchNorm1d(50))
        linear_1.add_module('drop_out', torch.nn.Dropout(0.3))
        self.linear_1 = linear_1
        linear_2 = nn.Sequential()
        linear_2.add_module('linear', nn.Linear(50, output_size))
        self.linear_2 = linear_2

    def forward(self, batch):
        x = [batch[n] for n in self.subscripted_views]
        x = [[self.transform(dgm) for dgm in view_batch] for view_batch in x]
        x = self.pht_sl(x)
        x = [l(xx) for l, xx in zip(self.stage_1, x)]
        x = [torch.squeeze(torch.max(xx, 1)[0]) for xx in x]
        x = [l(xx) for l, xx in zip(self.stage_2, x)]
        x = torch.cat(x, 1)
        x = self.linear_1(x)
        x = self.linear_2(x)
        return x

class MNISTAutoencoder(nn.Module):
    """MNIST Autoencoder from Original Paper Implementation."""
    def __init__(self, init_num_filters=16, lrelu_slope=0.2, inter_fc_dim=128, embedding_dim=2):
        super(MNISTAutoencoder, self).__init__()

        self.init_num_filters_ = init_num_filters
        self.lrelu_slope_ = lrelu_slope
        self.inter_fc_dim_ = inter_fc_dim
        self.embedding_dim_ = embedding_dim

        self.encoder = MNISTEncoder(init_num_filters, lrelu_slope, inter_fc_dim, embedding_dim)
        self.decoder = MNISTDecoder(init_num_filters, lrelu_slope, inter_fc_dim, embedding_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

class MNISTAdversariallearner(nn.Module):
    def __init__(self, subscripted_views=16):
        super(MNISTAdversariallearner, self).__init__()

        self.subscripted_views = 16

        self.discriminator = MNISTAdversariallearner(subscripted_views)

    def forward(self, batch):
        None
        return self.v
