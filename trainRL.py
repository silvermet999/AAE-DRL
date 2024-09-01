import collections

import numpy as np
import torch
import os
from torch import nn
import DDPG
import time
from AAE import main, AAE_archi
from torch.utils.data import Dataset, DataLoader
import utils
from torch.nn import BCELoss, L1Loss


class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.dataframe.iloc[idx]
        return sample

dataset = CustomDataset(main.X_test)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataloader,
                                               batch_size=32,
                                               num_workers=4,
                                               shuffle=False,
                                               pin_memory=True)


np.random.seed(5)
#torch.manual_seed(5)

def evaluate_policy(valid_loader,env, eval_episodes=6,render = False):
    avg_reward = 0.
    env.reset(epoch_size=len(valid_loader),figures=8) # reset the visdom and set number of figures

    #for i,(input) in enumerate(valid_loader):
    for i in range(0, eval_episodes):
        try:
            input = next(dataloader_iterator)
        except:
            dataloader_iterator = iter(valid_loader)
            input = next(dataloader_iterator)

       # data_iter = iter(valid_loader)
       # input = data_iter.next()
        #action_rand = torch.randn(args.batch_size, args.z_dim)
        obs =env.agent_input(input)# env(input, action_rand)
        done = False

        while not done:
          # Action By Agent and collect reward
            action = DDPG.DDPG.select_action #(np.array(obs))
            action= torch.tensor(action).cuda().unsqueeze(dim=0)
            new_state, _, reward, done, _ = env( input, action,render=render,disp =True)
            avg_reward += reward

        if i+1 >= eval_episodes:
            break

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print("---------------------------------------")

    return avg_reward

def test_policy(valid_loader,env, eval_episodes=12,render = True):
    avg_reward = 0.
    env.reset(epoch_size=len(valid_loader),figures=12) # reset the visdom and set number of figures

    #for i,(input) in enumerate(valid_loader):
    for i in range (0,eval_episodes):
        try:
            input = next(dataloader_iterator)
        except:
            dataloader_iterator = iter(valid_loader)
            input = next(dataloader_iterator)

       # data_iter = iter(valid_loader)
       # input = data_iter.next()
        #action_rand = torch.randn(args.batch_size, args.z_dim)
        obs =env.agent_input(input)# env(input, action_rand)
        done = False

        while not done:
          # Action By Agent and collect reward
            action = DDPG.DDPG.select_action #(np.array(obs))
            action= torch.tensor(action).cuda().unsqueeze(dim=0)
            new_state, _, reward, done, _ = env( input, action,render=render,disp =True)
            avg_reward += reward

        if i+1 >= eval_episodes:
            break;

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print("---------------------------------------")

    return avg_reward



def trainRL(train_loader,valid_loader,test_loader,model_encoder,model_decoder, model_D,epoch, chamfer,nll, mse,norm,vis_Valid,vis_Valida):

    model_encoder.eval()
    model_decoder.eval()
    model_D.eval()

    epoch_size = len(valid_loader)


    file_name = "1"

    env = envs(model_D, model_encoder, model_decoder, epoch_size)

    state_dim = 32
    action_dim = AAE_archi.z_dim
    max_action = 10000

    # Initialize policy

    policy = DDPG.DDPG(state_dim, action_dim, max_action)

    replay_buffer = utils.ReplayBuffer()

    evaluations = [evaluate_policy(valid_loader,env)]



    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True
    env.reset(epoch_size=len(train_loader))


    while total_timesteps < 100000:


        if done:

            try:
                input = next(dataloader_iterator)
            except:
                dataloader_iterator = iter(train_loader)
                input = next(dataloader_iterator)


            if total_timesteps != 0:
                # print("Total T: %d Episode Num: %d Episode T: %d Reward: %f") % (total_timesteps, episode_num, episode_timesteps, episode_reward)
                policy.train(replay_buffer, episode_timesteps, 32, 0.99, 0.005)

            # Evaluate episode
            if timesteps_since_eval >= 500:
                timesteps_since_eval %= 500

                evaluations.append(evaluate_policy(valid_loader,env, render = False))

                policy.save(file_name, directory="./pytorch_models")

                env.reset(epoch_size=len(test_loader))
                test_policy(test_loader, env, render=True)

                env.reset(epoch_size=len(train_loader))


            # Reset environment
           # obs = env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Select action randomly or according to policy
        obs = env.agent_input(input)

        if total_timesteps < 100000:
          #  action_t = torch.rand(args.batch_size, args.z_dim) # TODO checked rand instead of randn
            action_t = torch.FloatTensor(32, 32).uniform_(-max_action, max_action)
            action = action_t.detach().cpu().numpy().squeeze(0)



           # obs, _, _, _, _ = env(input, action_t)
        else:

           # action_rand = torch.randn(args.batch_size, args.z_dim)
           #
           # obs, _, _, _, _ = env( input, action_rand)

            action = policy.select_action(np.array(obs))
            if 0.1 != 0:
                action = (action + np.random.normal(0, 0.1, size=32)).clip(
                    max_action*np.ones(32,), max_action*np.ones(32,))
                action = np.float32(action)
            action_t = torch.tensor(action).cuda().unsqueeze(dim=0)
        # Perform action

        # env.render()

        new_obs, _, reward, done, _ = env(input, action_t,disp = True)

       # new_obs, reward, done, _ = env.step(action)
        done_bool = 0 if episode_timesteps + 1 == 5 else float(done)
        episode_reward += reward

        # Store data in replay buffer
        replay_buffer.add((obs, new_obs, action, reward, done_bool))

        obs = new_obs

        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1


    # for i,(input) in enumerate(valid_loader):
    #
    #
    #
    #      if np.shape(input)[0]< args.batch_size:
    #         break;#print(np.shape(input)[0])
    #
    #      action = torch.randn(args.batch_size, args.z_dim)
    #      action_np = action.detach().cpu().numpy()
    #      new_state, _, reward,done, _ = env1(i,input,action)
    #
    #
    #
    # return reward

class envs(nn.Module):
    def __init__(self,model_D,model_encoder,model_decoder,epoch_size):
        super(envs,self).__init__()

        self.bce = BCELoss()
        self.mae = L1Loss()
        self.epoch = 0
        self.epoch_size =epoch_size

        self.model_D = model_D
        self.model_encoder = model_encoder
        self.model_decoder = model_decoder
        self.j = 1
        self.figures = 3
        self.attempts = 5
        self.end = time.time()
        self.batch_time = utils.AverageMeter()
        self.lossess = utils.AverageMeter()
        self.attempt_id =0
        self.state_prev = np.zeros([4,])
        self.iter = 0
    def reset(self,epoch_size,figures =3):
        self.j = 1
        self.i = 0
        self.figures = figures
        self.epoch_size= epoch_size
    def agent_input(self,input):
        with torch.no_grad():
            input = input.cuda()
            input_var = torch.Variable(input, requires_grad=True)
            encoder_out = self.model_encoder(input_var, )
            out = encoder_out.detach().cpu().numpy().squeeze()
        return out

    def forward(self,input,action,disp=False):
        with torch.no_grad():
            # Encoder Input
            input = input.cuda()
            input_var = torch.Variable(input, requires_grad=True)

            # Encoder  output
            encoder_out = self.model_encoder(input_var, )

            # D Decoder Output
#            pc_1, pc_2, pc_3 = self.model_decoder(encoder_out)
            pc_1 = self.model_decoder(encoder_out)
            # Generator Input
            z = torch.Variable(action, requires_grad=True).cuda()

            # Generator Output
            out_GD, _ = self.model_G(z)
            out_G = torch.squeeze(out_GD, dim=1)
            out_G = out_G.contiguous().view(-1, 32)

            # Discriminator Output
            #out_D, _ = self.model_D(encoder_out.view(-1,1,32,32))
        #    out_D, _ = self.model_D(encoder_out.view(-1, 1, 1,args.state_dim)) # TODO Alert major mistake
            out_D, _ = self.model_D(out_GD) # TODO Alert major mistake

            # H Decoder Output
#            pc_1_G, pc_2_G, pc_3_G = self.model_decoder(out_G)
            pc_1_G = self.model_decoder(out_G)


            # Preprocesing of Input PC and Predicted PC for Visdom
            trans_input = torch.squeeze(input_var, dim=1)
            trans_input = torch.transpose(trans_input, 1, 2)
            trans_input_temp = trans_input[0, :, :]
            pc_1_temp = pc_1[0, :, :] # D Decoder PC
            pc_1_G_temp = pc_1_G[0, :, :] # H Decoder PC


        # Discriminator Loss
        loss_D = self.nll(out_D)

        # Loss Between Noisy GFV and Clean GFV
        loss_GFV = self.mse(out_G, encoder_out)

        # Norm Loss
        loss_norm = self.norm(z)

        # Chamfer loss
        loss_chamfer = self.chamfer(pc_1_G, pc_1)  # #self.chamfer(pc_1_G, trans_input) instantaneous loss of batch items

        # States Formulation
        state_curr = np.array([loss_D.cpu().data.numpy(), loss_GFV.cpu().data.numpy()
                                  , loss_chamfer.cpu().data.numpy(), loss_norm.cpu().data.numpy()])
      #  state_prev = self.state_prev

        reward_D = state_curr[0]#state_curr[0] - self.state_prev[0]
        reward_GFV =-state_curr[1]# -state_curr[1] + self.state_prev[1]
        reward_chamfer = -state_curr[2]#-state_curr[2] + self.state_prev[2]
        reward_norm =-state_curr[3] # - state_curr[3] + self.state_prev[3]
        # Reward Formulation
        reward = ( reward_D *0.01 + reward_GFV * 10.0 + reward_chamfer *100.0 + reward_norm*1/10)#( reward_D + reward_GFV * 10.0 + reward_chamfer *100 + reward_norm*0.002) #reward_GFV + reward_chamfer + reward_D * (1/30)  TODO reward_D *0.002 + reward_GFV * 10.0 + reward_chamfer *100 + reward_norm  ( reward_D *0.2 + reward_GFV * 100.0 + reward_chamfer *100 + reward_norm)
      #  reward = reward * 100
     #   self.state_prev = state_curr

        #self.lossess.update(loss_chamfer.item(), input.size(0))  # loss and batch size as input

        # measured elapsed time
        self.batch_time.update(time.time() - self.end)
        self.end = time.time()

       # if i % args.print_freq == 0 :

      #  if self.j <= 5:
        visuals = collections.OrderedDict(
            [('Input_pc', trans_input_temp.detach().cpu().numpy()),
             ('AE Predicted_pc', pc_1_temp.detach().cpu().numpy()),
             ('GAN Generated_pc', pc_1_G_temp.detach().cpu().numpy())])

        if disp:
            print('[{4}][{0}/{1}]\t Reward: {2}\t States: {3}'.format(self.i, self.epoch_size,reward,state_curr,self.iter))
            self.i += 1
            if(self.i>=self.epoch_size):
                self.i=0
                self.iter +=1


      #  errors = OrderedDict([('loss', loss_chamfer.item())])  # plotting average loss
     #   vis_Valid.plot_current_errors(self.epoch, float(i) / self.epoch_size, args, errors)
        # if self.attempt_id ==self.attempts:
        #     done = True
        # else :
        #     done = False
        done = True
        state = out_G.detach().cpu().data.numpy().squeeze()
        return state, _, reward, done, self.lossess.avg









