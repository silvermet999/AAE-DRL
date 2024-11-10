import torch
import torch.utils.data
import torch.nn.parallel
from EnvClass import Env

import os

from utils import RL_dataloader, display_env
from RL import TD3


class Tester(object):
    """RL tester"""

    def __init__(self, args, test_loader, model_encoder, model_decoder, model_g, model_d,
                 model_classifier):
        """
        initialize RL trainer

        Parameters
        ----------
        args : dict
            args dictionary look at :func: '~gan_main.parse_args'
        test_loader : torch.utils.data.dataloader
            torch data loader of test dataset
        model_encoder : torch.nn.Module
            encoder model
        model_decoder : torch.nn.Module
            decoder model
        model_g : torch.nn.Module
            generator model
        model_d : torch.nn.Module
            discriminator model
        """
        self.device = args.device

        self.test_loader = RL_dataloader(test_loader)

        self.batch_size = args.batch_size
        self.save_models = args.save_models
        self.max_episodes_steps = args.max_episodes_steps

        self.z_dim = args.z_dim
        self.max_action = args.max_action

        self.encoder = model_encoder
        self.decoder = model_decoder
        self.G = model_g
        self.D = model_d
        self.model_classifier = model_classifier

        self.env = Env(args, self.G, self.D, self.model_classifier, self.decoder)

        self.state_dim = args.state_dim
        self.action_dim = args.z_dim
        self.max_action = args.max_action

        self.policy = TD3(self.device, self.state_dim, self.action_dim, self.max_action)

        self.model_path = os.path.join(args.model_dir, 'RL_train')
        self.save_path = os.path.join(args.result_dir, 'RL_test')
        os.makedirs(self.save_path, exist_ok=True)
        
        self.policy.load(args.model_name, directory=self.model_path)

    def evaluate(self):
        """
        evaluate RL
        """
        # name of result for saving
        episode_num = 0
        number_correct = 0
        while True:

            # if loader present create until data finished, else random generation of input data
            print('input loader')
            try:
                state_t, label = self.test_loader.next_data()
                episode_target = (torch.randint(10, label.shape) + label) % 10
                state = self.env.set_state(state_t)
                done = False
                episode_return = 0
            except:
                break

            while not done:
                # ================== Run RL ================== #
                with torch.no_grad():
                    action = self.policy.select_action(state, episode_target)
                    action_t = torch.tensor(action).to(self.device).unsqueeze(dim=0)
                    # Perform action
                    next_state, reward, done, c = self.env(action_t, episode_target)

                state = next_state
                episode_return += reward.mean()
                for i, r in enumerate(reward):
                    with open(os.path.join(self.save_path, "logs.txt"), "a") as f:
                        f.write('{} \n {} {} \n'.format(r, episode_target[i].data,  c[i]))
                        f.close()

                    with open(os.path.join(self.save_path, "result.txt"), "w") as f:
                        if r > 20.0:
                            number_correct += 1
                        f.write('{}\n'.format(number_correct))
                        f.close()
            print('\repisode: {}, reward: {}'.format(episode_num + 1, episode_return))
            episode_num += 1

            self.env.reset()

            with torch.no_grad():
                #out_state = torch.FloatTensor(state.reshape((1, 1, -1))).to(self.device)
                print(state_t.shape, state.shape)
                state_image = self.decoder(state_t)
                out_state_image = self.decoder(state)
            for i, s in enumerate(state_image):
                display_env(state_image[i], out_state_image[i], reward[i],
                        os.path.join(self.save_path, "episode_{}_{}".format(episode_num, i)),
                        target=episode_target[i].detach().cpu().numpy())

            yield state_image, out_state_image, episode_return
