from torch.utils.data import DataLoader
from data import main_u
from torch.nn import Linear, Softmax, Tanh, Sequential, Module, ModuleDict
import torch
import numpy as np
from AAE import AAE_archi_opt
from torchmetrics.functional.classification import multiclass_accuracy, binary_accuracy


# y = pd.get_dummies(y).astype(int)
# cols = ['Normal', 'Backdoor', 'Analysis', 'Fuzzers', 'Shellcode',
#        'Reconnaissance', 'Exploits', 'DoS', 'Worms', 'Generic']
# y = y.rename(columns = dict(zip(y.columns, cols)))


# X_train = np.loadtxt('/home/silver/PycharmProjects/AAEDRL/AAE/Adam.txt')
df_filtered = main_u.df[main_u.df["attack_cat"] != 6]
X_train, X_test, y_train, y_test = main_u.vertical_split(main_u.corr(df_filtered.drop(["attack_cat", "label"], axis=1)),
                                                    df_filtered["attack_cat"])

X_train_sc = main_u.min_max(X_train)
y_train = y_train.to_numpy()


dataset = AAE_archi_opt.CustomDataset(X_train_sc, y_train)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

class Classifier(Module):
    def __init__(self, discrete_features, continuous_features):
        super(Classifier, self).__init__()

        self.discrete_features = discrete_features
        self.continuous_features = continuous_features

        self.shared = Sequential(
        Linear(21, 15),
        Linear(15, 9)
        )

        self.discrete_out = {feature: Linear(9, num_classes)
                             for feature, num_classes in enumerate(self.discrete_features)}
        self.continuous_out = {feature: Linear(9, 1)
                               for feature in range(len(self.continuous_features))}

        # self.discrete_out = ModuleDict(self.discrete_out)
        # self.continuous_out = ModuleDict(self.continuous_out)

        self.softmax = Softmax(dim=-1)
        self.tanh = Tanh()

    def forward(self, x):
        shared_features = self.shared(x)

        discrete_outputs = {}
        continuous_outputs = {}

        for i in range(len(self.discrete_features)):
            logits = self.discrete_out[i](shared_features)
            discrete_outputs[i] = self.softmax(logits)

        for i in range(len(self.continuous_features)):
            continuous_outputs[i] = self.tanh(self.continuous_out[i](shared_features))

        discrete_values = list(discrete_outputs.values())
        discrete_fs = torch.cat(discrete_values, dim=1)

        continuous_values = list(continuous_outputs.values())
        continuous_fs = torch.cat(continuous_values, dim=1)

        return discrete_fs, continuous_fs

    def calc_accuracy(self, outputs, targets):
        discrete_outputs, continuous_outputs = outputs
        discrete_targets, continuous_targets = targets
        total_accuracy = 0
        num_features = 0

        for feature in self.discrete_features:
            if feature in discrete_targets:
                _, predicted_labels = torch.max(discrete_outputs[feature], 1)
                correct_predictions = (predicted_labels == discrete_targets[feature]).sum().item()
                total_accuracy += correct_predictions / discrete_targets[feature].size(0)
                num_features += 1

        for feature in self.continuous_features:
            if feature in continuous_targets:
                predicted_labels = (continuous_outputs[feature] >= 0.5).float()
                correct_predictions = (predicted_labels == continuous_targets[feature]).sum().item()
                total_accuracy += correct_predictions / continuous_targets[feature].size(0)
                num_features += 1

        return total_accuracy / num_features if num_features > 0 else 0

discrete = {2: 132,
            3: 13,
            4: 7,
            23: 4,
            24: 11,
            25: 16
            }
continuous = [0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

classifier = Classifier(discrete, continuous)
class_opt = torch.optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.99))
Tensor = torch.FloatTensor

for epoch in range(50):
    for i, (X, y) in enumerate(dataloader):
        fs = X.type(Tensor)
        discrete_targets = {}
        continuous_targets = {}
        for feature, _ in classifier.discrete_features.items():
            discrete_targets[feature] = fs[feature]

        for feature in classifier.continuous_features:
            continuous_targets[feature] = fs[feature]


        class_opt.zero_grad()
        class_pred_disc, class_pred_cont = classifier(fs)
        class_acc = classifier.calc_accuracy((class_pred_disc, discrete_targets), (class_pred_cont,
                                                                                   continuous_targets))

        class_acc.backward()
        class_opt.step()
        print(class_acc.item())
