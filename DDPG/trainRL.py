import numpy as np
import torch
from torch import nn
import DDPG
import time
from AAE import AAE_archi
from data import main
import utils

cuda = True if torch.cuda.is_available() else False

# class Norm(nn.Module):
#     def __init__(self,dims):
#         super(Norm,self).__init__()
#         self.dims =dims
#
#     def forward(self,x):
#         z2 = torch.norm(x,p=2)
#         out = (z2-self.dims)
#         out = out*out
#         return out

class NLL(nn.Module):
    def __init__(self):
        super(NLL,self).__init__()
    def forward(self,x):
        return torch.mean(x)



np.random.seed(5)

def evaluate_policy(env, eval_episodes=6,render = False):
    avg_reward = 0.
    env.reset()
    for i in range(0, eval_episodes):
        df_iterator = main.X_train.itertuples(index=False, name=None)
        input = next(df_iterator)
        print(np.shape(input)) # (105,)
        obs =env.agent_input(input)
        done = False

        while not done:
          # Action By Agent and collect reward
            action = DDPG.DDPG.select_action(np.array(obs))
            action= torch.tensor(action).cuda() if cuda else torch.tensor(action)
            new_state, _, reward, done, _ = env( input, action, render=render, disp =True)
            avg_reward += reward

        if i+1 >= eval_episodes:
            break

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print("---------------------------------------")

    return avg_reward

def test_policy(env, eval_episodes=12,render = True):
    avg_reward = 0.
    env.reset()
    for i in range (0,eval_episodes):
        df_iterator = main.X_train.itertuples(index=False, name=None)
        input = next(df_iterator)



        # obs =env.agent_input(input)# env(input, action_rand)
        done = False

        while not done:
          # Action By Agent and collect reward
            action = DDPG.DDPG.select_action #(np.array(obs))
            action= torch.tensor(action).cuda().unsqueeze(dim=0) if cuda else torch.tensor(action).unsqueeze(dim=0)
            new_state, _, reward, done, _ = env( input, action,render=render,disp =True)
            avg_reward += reward

        if i+1 >= eval_episodes:
            break

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print("---------------------------------------")

    return avg_reward



def trainRL():

    AAE_archi.encoder_generator.eval()
    AAE_archi.discriminator.eval()



    file_name = "1"

    env = envs()

    state_dim = 46
    action_dim = AAE_archi.z_dim
    max_action = 10000

    # Initialize policy

    policy = DDPG.DDPG(state_dim, action_dim, max_action)

    replay_buffer = utils.ReplayBuffer()

    evaluations = [evaluate_policy(env)]



    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True
    env.reset()


    while total_timesteps < 100000:


        if done:
            df_iterator = main.X_train.itertuples(index=False, name=None)
            input = next(df_iterator)

            if total_timesteps != 0:
                policy.train(replay_buffer, episode_timesteps, 32, 0.99, 0.005)

            # Evaluate episode
            if timesteps_since_eval >= 500:
                timesteps_since_eval %= 500

                evaluations.append(evaluate_policy(env, render = False))

                policy.save(file_name, directory="./pytorch_models")

                env.reset()
                test_policy(env, render=True)

                env.reset()

            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Select action randomly or according to policy
        obs = env.agent_input(input)

        if total_timesteps < 100000:
            action_t = torch.FloatTensor(32, 32).uniform_(-max_action, max_action)
            action = action_t.detach().cpu().numpy().squeeze(0)

        else:
            action = DDPG.DDPG.select_action(np.array(obs))
            if 0.1 != 0:
                action = (action + np.random.normal(0, 0.1, size=32)).clip(
                    max_action*np.ones(32,), max_action*np.ones(32,))
                action = np.float32(action)
            action_t = torch.tensor(action).cuda().unsqueeze(dim=0) if cuda else torch.tensor(action).unsqueeze(dim=0)

        new_obs, _, reward, done, _ = env(input, action_t,disp = True)

        done_bool = 0 if episode_timesteps + 1 == 5 else float(done)
        episode_reward += reward

        # Store data in replay buffer
        replay_buffer.add((obs, new_obs, action, reward, done_bool))

        # obs = new_obs

        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1


class envs(nn.Module):
    def __init__(self):
        super(envs,self).__init__()

        self.nll = NLL()
        self.epoch = 0
        self.epoch_size = 100
        self.model_e = AAE_archi.encoder_generator

        self.model_D = AAE_archi.discriminator
        self.j = 1
        self.figures = 3
        self.attempts = 5
        self.end = time.time()
        self.batch_time = utils.AverageMeter()
        self.lossess = utils.AverageMeter()
        self.attempt_id =0
        self.state_prev = np.zeros([4,])
        self.iter = 0
    def reset(self,figures =3):
        self.j = 1
        self.i = 0
        self.figures = figures
        self.epoch_size= 100
    def agent_input(self,input):
        with torch.no_grad():
            input_var = torch.tensor(input, dtype=torch.float32).cuda() if cuda else torch.tensor(input, dtype=torch.float32)
            input_var = np.array(input_var)
            input_var = input_var.reshape(1, -1)
            print(np.shape(input_var))
            encoder_out = AAE_archi.encoder_generator(input_var)

            out = encoder_out.detach().cpu().numpy().squeeze()
        return out

    def forward(self,input,action, disp=False):
        with torch.no_grad():
            # Encoder Input
            input_var = torch.tensor(input).cuda() if cuda else torch.tensor(input)

            # Encoder  output
            encoder_out = AAE_archi.encoder_generator(input_var)
            z = torch.tensor(action).cuda() if cuda else torch.tensor(action)

            # Discriminator Output
            out_D = self.model_D(encoder_out) # TODO Alert major mistake



        # Discriminator Loss
        loss_D = self.nll(out_D)


        # Norm Loss
        loss_norm = nn.BatchNorm1d(z)

        state_curr = np.array([loss_D.cpu().data.numpy(), loss_norm.cpu().data.numpy()])

        reward_D = state_curr[0]
        reward_norm =-state_curr[3]
        # Reward Formulation
        reward = (reward_D *0.01 + reward_norm*1/10)


        # measured elapsed time
        self.batch_time.update(time.time() - self.end)
        self.end = time.time()

        if disp:
            print('[{4}][{0}/{1}]\t Reward: {2}\t States: {3}'.format(self.i, self.epoch_size,reward,state_curr,self.iter))
            self.i += 1
            if(self.i>=self.epoch_size):
                self.i=0
                self.iter +=1
        done = True
        state = encoder_out.detach().cpu().data.numpy().squeeze()
        return state, reward, done, self.lossess.avg


train = trainRL()






