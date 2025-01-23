import csv

import torch
import torch.utils.data
from EnvClass import Env
from AAE import AAE_archi_opt
from clfs import classifier


from utils import RL_dataloader
from RL import TD3


class Tester(object):
    def __init__(self, test_loader, model_encoder, model_De, model_D, classifier, discrete):


        self.test_loader = RL_dataloader(test_loader)

        self.batch_size = 2
        self.max_episodes_steps = 100000

        self.max_action = 1

        self.encoder = model_encoder
        self.De = model_De
        self.D = model_D
        self.classifier = classifier

        self.env = Env(self.encoder, self.D, self.De, self.classifier)

        self.state_dim = 30
        self.action_dim = 5
        self.discrete_features = discrete
        self.max_action = 1
        self.policy = TD3(self.state_dim, self.action_dim, self.discrete_features, self.max_action)



    def evaluate(self):
        episode_num = 0
        number_correct = 0
        while True:
            print('input loader')
            try:
                state_t, label = self.test_loader.next_data()
                episode_target = (torch.randint(4, label.shape) + label) % 4
                state = self.env.set_state(state_t)
                done = False
                episode_return = 0
            except:
                break

            while not done:
                with torch.no_grad():
                    continuous_act, discrete_act = self.policy.select_action(state)
                    next_state, reward, done = self.env(continuous_act, discrete_act, episode_target)

                state = next_state
                episode_return += reward.mean()
            print('\repisode: {}, reward: {}'.format(episode_num + 1, episode_return))
            episode_num += 1

            self.env.reset()

            with torch.no_grad():
                new_state = self.encoder(torch.tensor(state))
                nd, nc, nb = self.De(new_state)
            nd = {key: tensor.detach().cpu().numpy() for key, tensor in nd.items()}
            nc = {key: tensor.detach().cpu().numpy() for key, tensor in nc.items()}
            nb = {key: tensor.detach().cpu().numpy() for key, tensor in nb.items()}

            with open('discrete.csv', 'w', newline='') as file_d:
                writer = csv.writer(file_d)
                for key, value in nd.items():
                    writer.writerow([key, value])

            with open('continuous.csv', 'w', newline='') as file_c:
                writer = csv.writer(file_c)
                for key, value in nc.items():
                    writer.writerow([key, value])

            with open('binary.csv', 'w', newline='') as file_b:
                writer = csv.writer(file_b)
                for key, value in nb.items():
                    writer.writerow([key, value])

            yield nd, nc, nb, episode_return

enc_gen = AAE_archi_opt.encoder_generator
test_loader = AAE_archi_opt.dataset_function(AAE_archi_opt.dataset, 32, 32, train=False)
decoder = AAE_archi_opt.Decoder(10, AAE_archi_opt.discrete, AAE_archi_opt.continuous, AAE_archi_opt.binary)
d_model = AAE_archi_opt.discriminator
classifier = classifier.classifier
d_model.eval()
discrete = {"state": 5,
            # "service": 13,
            "ct_state_ttl": 6,
            # "dttl": 9,
            # "sttl": 13,
            "trans_depth": 11,
            "proto": 2,
            "is_ftp_login":2
}
tester = Tester(test_loader, enc_gen, decoder, d_model, classifier, discrete)
evaluater = tester.evaluate()
for i in range(100):
    next(evaluater)
