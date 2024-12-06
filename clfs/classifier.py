"""Sparsemax activation function.

Pytorch implementation of Sparsemax function from:
-- "From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification"
-- André F. T. Martins, Ramón Fernandez Astudillo (http://arxiv.org/abs/1602.02068)

credits: https://github.com/KrisKorrel/sparsemax-pytorch/blob/master/sparsemax.py

"""

from __future__ import division

import torch
import torch.nn as nn
import numpy as np
from data import main_u

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Sparsemax(nn.Module):
    """Sparsemax function."""

    def __init__(self, dim=None):
        """Initialize sparsemax activation

        Args:
            dim (int, optional): The dimension over which to apply the sparsemax function.
        """
        super(Sparsemax, self).__init__()

        self.dim = -1 if dim is None else dim

    def forward(self, input):
        """Forward function.

        Args:
            input (torch.Tensor): Input tensor. First dimension should be the batch size

        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor

        """
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape to a convenient shape and reshape back after sparsemax
        input = input.transpose(0, self.dim)
        original_size = input.size()
        input = input.reshape(input.size(0), -1)
        input = input.transpose(0, 1)
        dim = 1

        number_of_logits = input.size(dim)

        # Translate input by max for numerical stability
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        # Sort input in descending order.
        # (NOTE: Can be replaced with linear time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.arange(start=1, end=number_of_logits + 1, step=1, device=device, dtype=input.dtype).view(1, -1)
        range = range.expand_as(zs)

        # Determine sparsity of projection
        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        # Sparsemax
        self.output = torch.max(torch.zeros_like(input), input - taus)

        # Reshape back to original shape
        output = self.output
        output = output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, self.dim)

        return output

    def backward(self, grad_output):
        """Backward function."""
        dim = 1

        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))

        return self.grad_input

sparsemax = Sparsemax(dim=1)


# GLU
def glu(act, n_units):
    act[:, :n_units] = act[:, :n_units].clone() * torch.nn.Sigmoid()(act[:, n_units:].clone())

    return act


class TabNetModel(nn.Module):

    def __init__(
            self,
            columns=3,
            num_features=26,
            feature_dims=80,
            output_dim=64,
            num_decision_steps=6,
            relaxation_factor=0.5,
            batch_momentum=0.001,
            virtual_batch_size=2,
            num_classes=2,
            epsilon=0.00001
    ):

        super().__init__()

        self.columns = columns
        self.num_features = num_features
        self.feature_dims = feature_dims
        self.output_dim = output_dim
        self.num_decision_steps = num_decision_steps
        self.relaxation_factor = relaxation_factor
        self.batch_momentum = batch_momentum
        self.virtual_batch_size = virtual_batch_size
        self.num_classes = num_classes
        self.epsilon = epsilon

        self.feature_transform_linear1 = torch.nn.Linear(num_features, self.feature_dims * 2, bias=False)
        self.BN = torch.nn.BatchNorm1d(num_features, momentum=batch_momentum)
        self.BN1 = torch.nn.BatchNorm1d(self.feature_dims * 2, momentum=batch_momentum)

        self.feature_transform_linear2 = torch.nn.Linear(self.feature_dims * 2, self.feature_dims * 2, bias=False)
        self.feature_transform_linear3 = torch.nn.Linear(self.feature_dims * 2, self.feature_dims * 2, bias=False)
        self.feature_transform_linear4 = torch.nn.Linear(self.feature_dims * 2, self.feature_dims * 2, bias=False)

        self.mask_linear_layer = torch.nn.Linear(self.feature_dims * 2 - output_dim, self.num_features, bias=False)
        self.BN2 = torch.nn.BatchNorm1d(self.num_features, momentum=batch_momentum)

        self.final_classifier_layer = torch.nn.Linear(self.output_dim, self.num_classes, bias=False)

    def encoder(self, data):

        batch_size = data.shape[0]
        features = self.BN(data)
        output_aggregated = torch.zeros([batch_size, self.output_dim])

        masked_features = features
        mask_values = torch.zeros([batch_size, self.num_features])

        aggregated_mask_values = torch.zeros([batch_size, self.num_features])
        complemantary_aggregated_mask_values = torch.ones([batch_size, self.num_features])

        total_entropy = 0

        for ni in range(self.num_decision_steps):

            if ni == 0:

                transform_f1 = self.feature_transform_linear1(masked_features)
                norm_transform_f1 = self.BN1(transform_f1)

                transform_f2 = self.feature_transform_linear2(norm_transform_f1)
                norm_transform_f2 = self.BN1(transform_f2)

            else:

                transform_f1 = self.feature_transform_linear1(masked_features)
                norm_transform_f1 = self.BN1(transform_f1)

                transform_f2 = self.feature_transform_linear2(norm_transform_f1)
                norm_transform_f2 = self.BN1(transform_f2)

                # GLU
                transform_f2 = (glu(norm_transform_f2, self.feature_dims) + transform_f1) * np.sqrt(0.5)

                transform_f3 = self.feature_transform_linear3(transform_f2)
                norm_transform_f3 = self.BN1(transform_f3)

                transform_f4 = self.feature_transform_linear4(norm_transform_f3)
                norm_transform_f4 = self.BN1(transform_f4)

                # GLU
                transform_f4 = (glu(norm_transform_f4, self.feature_dims) + transform_f3) * np.sqrt(0.5)

                decision_out = torch.nn.ReLU(inplace=True)(transform_f4[:, :self.output_dim])
                # Decision aggregation
                output_aggregated = torch.add(decision_out, output_aggregated)
                scale_agg = torch.sum(decision_out, axis=1, keepdim=True) / (self.num_decision_steps - 1)
                aggregated_mask_values = torch.add(aggregated_mask_values, mask_values * scale_agg)

                features_for_coef = (transform_f4[:, self.output_dim:])

                if ni < (self.num_decision_steps - 1):
                    mask_linear_layer = self.mask_linear_layer(features_for_coef)
                    mask_linear_norm = self.BN2(mask_linear_layer)
                    mask_linear_norm = torch.mul(mask_linear_norm, complemantary_aggregated_mask_values)
                    mask_values = sparsemax(mask_linear_norm)

                    complemantary_aggregated_mask_values = torch.mul(complemantary_aggregated_mask_values,
                                                                     self.relaxation_factor - mask_values)
                    total_entropy = torch.add(total_entropy, torch.mean(
                        torch.sum(-mask_values * torch.log(mask_values + self.epsilon), axis=1)) / (
                                                          self.num_decision_steps - 1))
                    masked_features = torch.mul(mask_values, features)

        return output_aggregated, total_entropy

    def classify(self, output_logits):

        logits = self.final_classifier_layer(output_logits)
        predictions = torch.nn.Softmax(dim=1)(logits)

        return logits, predictions

df_filtered = main_u.df[main_u.df["attack_cat"].isin([5, 3])]
X_train, X_test, y_train, y_test = main_u.vertical_split(main_u.corr(df_filtered.drop(["attack_cat", "label"], axis=1)),
                                                    df_filtered["attack_cat"])


X_train_disc = X_train[["proto", "service", "state", "is_ftp_login", "ct_flw_http_mthd"]].to_numpy()
X_train_cont = main_u.mac(X_train[[feature for feature in X_train.columns if feature not in X_train_disc]])
X_train_sc = np.concatenate((X_train_disc, X_train_cont), axis=1)

X_test_disc = X_test[["proto", "service", "state", "is_ftp_login", "ct_flw_http_mthd"]].to_numpy()
X_test_cont = main_u.mac(X_test[[feature for feature in X_test.columns if feature not in X_test_disc]])
X_test_sc = np.concatenate((X_test_disc, X_test_cont), axis=1)

y_train = y_train.to_numpy()

classifier = TabNetModel()
encoded = classifier.encoder(torch.tensor(X_train_sc).float().to(device))
logit, class_pred = classifier.classify(torch.tensor(encoded[0]))
