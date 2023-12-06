

import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from make_env import make_env
from pettingzoo.mpe import simple_adversary_v2

##多智能体buffer存储所有智能体动作，前后状态，奖励，buffer大小，采样的batch都是类基本参数
class MultiAgentReplayBuffer:
    def __init__(self,buf_size,lagents_obs_dim,lagents_action_dim,agents_num,batch_size):
        self.buf_size=buf_size
        self.buf_cntr=0
        self.agents_num=agents_num
        self.batch_size=batch_size
        self.lagents_obs_dim=lagents_obs_dim
        self.lagents_action_dim=lagents_action_dim
        self.reward_buf=np.zeros((self.buf_size,agents_num))
        self.terminal_buf=np.zeros((self.buf_size,agents_num),dtype=bool)   
        self.init_actor_buf() 
    ##用list来构建多个智能体的buffer，初始化存储空间
    def init_actor_buf(self):
        self.lagents_state_buf=[]
        self.lagents_new_state_buf=[]
        self.lagents_action_buf=[]
        for i in range(self.agents_num):
            self.lagents_state_buf.append(np.zeros((self.buf_size,self.lagents_obs_dim[i])))
            self.lagents_new_state_buf.append(np.zeros((self.buf_size,self.lagents_obs_dim[i])))
            self.lagents_action_buf.append(np.zeros((self.buf_size,self.lagents_action_dim[i])))
    ##往存储空间存数据
    def store_transition(self,agents_obs,action,reward,agents_new_obs,done):
        if self.buf_cntr % self.buf_size ==0 and self.buf_cntr >0:
            self.init_actor_buf
        index = self.buf_cntr % self.buf_size
        for agent_idx in range(self.agents_num):
            self.lagents_state_buf[agent_idx][index] = agents_obs[agent_idx]
            self.lagents_new_state_buf[agent_idx][index] = agents_new_obs[agent_idx]
            self.lagents_action_buf[agent_idx][index] = action[agent_idx]

        self.reward_buf[index] = reward
        self.terminal_buf[index] = done
        self.buf_cntr +=1
    ## 从buffer采样batch个样本
    def sample_buffer(self):
        buf_size = min(self.buf_cntr,self.buf_size)
        batch = np.random.choice(buf_size, self.batch_size, replace=False)
        rewards = self.reward_buf[batch]
        terminal = self.terminal_buf[batch]
        lagents_states = []
        lagents_new_states =[]
        lagents_actions = []
        for agent_idx in range(self.agents_num):
            lagents_states.append(self.lagents_state_buf[agent_idx][batch])
            lagents_new_states.append(self.lagents_new_state_buf[agent_idx][batch])
            lagents_actions.append(self.lagents_action_buf[agent_idx][batch])

        return lagents_states, lagents_actions, rewards, lagents_new_states, terminal 
    ## buffer足够多样本后再允许采样
    def ready(self):
        if self.buf_cntr >= self.batch_size:
            return True
        return False
## critic网络类，就是个深度网络，确定好网络规模，网络优化器及学习率等
class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dim, fc1_dim, fc2_dim, output_dim, name, chkpt_dir):
        super(CriticNetwork, self).__init__()

        ##############参数保存文件名########################
        self.chkpt_file = os.path.join(chkpt_dir, name).replace('\\','/')
        #### 网络输入层-中间层-输出层维数的确定
        self.fc1 = nn.Linear(input_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.y = nn.Linear(fc2_dim,output_dim)
        #### 优化器的确定
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cude:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    ## 网络前向计算
    def forward(self, state, action):
        x = F.relu(self.fc1(T.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))
        y = self.y(x)
        return y
    ## 保存和加载网络参数
    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)
    def load_chckpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))

## actor网络类，就是个深度网络，确定好网络规模，网络优化器及学习率等
class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dim, fc1_dim, fc2_dim, output_dim, name, chkpt_dir):
        super(ActorNetwork, self).__init__()

        ##############参数保存文件名########################
        self.chkpt_file = os.path.join(chkpt_dir, name).replace('\\','/')
        #print(self.chkpt_file)
        ##############################################
        #### 网络输入层-中间层-输出层维数的确定
        self.fc1 = nn.Linear(input_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.y = nn.Linear(fc2_dim, output_dim)
        #### 优化器的确定
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cude:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    ## 网络前向计算
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        y = T.softmax(self.y(x), dim=1)
        return y
    ## 保存和加载网络参数
    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file) 
    def load_chckpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))



## agent类，每个agent四个网络来刻画，执行动作等基本操作

class Agent:
    def __init__(self, critic_input_dim, critic_output_dim, actor_input_dim,  actor_output_dim, agent_idx, alpha, beta, fc1, fc2,  tau, chkpt_dir):
        ## 基本参数
        self.tau = tau # 更新target网络中的参数
        self.agent_name = 'agent_%s' % agent_idx # agent名

        ##### 创建agent的四个网络#####
        self.actor = ActorNetwork(alpha, actor_input_dim, fc1, fc2, actor_output_dim, chkpt_dir=chkpt_dir, name=self.agent_name+'_actor')

        self.critic = CriticNetwork(beta, critic_input_dim, fc1, fc2, critic_output_dim, chkpt_dir=chkpt_dir, name=self.agent_name+'_critic')

        self.target_actor = ActorNetwork(alpha, actor_input_dim, fc1, fc2, actor_output_dim, chkpt_dir=chkpt_dir, name=self.agent_name+'_target_actor')

        self.target_critic = CriticNetwork(beta, critic_input_dim, fc1, fc2, critic_output_dim, chkpt_dir=chkpt_dir, name=self.agent_name+'_target_critic') 

    ##### 创建agent的四个网络#####
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()
        target_actor_params_dict = dict(target_actor_params)
        actor_params_dict = dict(actor_params)

        for name in actor_params_dict:
            target_actor_params_dict[name] = tau*actor_params_dict[name].clone() + (1-tau)*target_actor_params_dict[name].clone()
        ##更新target_actor网络参数
        self.target_actor.load_state_dict(target_actor_params_dict)

        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()
        target_critic_params_dict = dict(target_critic_params)
        critic_params_dict = dict(critic_params)
        ##clone使用与不用的区别，输出print(critic_params_dict[name])看看情况
        for name in critic_params_dict:
            target_critic_params_dict[name] = tau*critic_params_dict[name].clone() + (1-tau)*target_critic_params_dict[name].clone()
        ##更新target_critic网络参数
        self.target_critic.load_state_dict(target_critic_params_dict)

    ###选择动作,MPE环境于pettingzoo环境不一样
    def choose_action(self, observation, explore =False):
        #numpy的observation转tensor
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        action = self.actor.forward(state)
        # print('hello action',action)
        if explore:
            action = sample_with_gumbel_softmax(action) ### gumbel_softmax采样输出
            action = distribution_to_onehot(action)### 独热码输出
        else:
                action = distribution_to_onehot(action) ### 轮盘法输出
        return action.detach().cpu().numpy()[0]

    ###保存网络
    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()
    def load_models(self):
        self.actor.load_chckpoint()
        self.target_actor.load_chckpoint()
        self.critic.load_chckpoint()
        self.target_critic.load_chckpoint()

## multi_agent类，构造agent，执行动作等基本操作
class MADDPG:
    def __init__(self, lagents_obs_dim, lagents_action_dim, agents_num, scenario='simple', alpha=0.01, beta=0.01, fc1=64, fc2=64, gamma=0.99, tau=0.01, chkpt_dir='/chkpt'):
        self.agents = []
        self.agents_num = agents_num
        self.gamma = gamma
        self.scenario = scenario
        ## 确定每个agent网络的输入、输出维数，用list存储
        self.agents_critic_input_dim = []
        self.agents_critic_output_dim = []
        self.agents_actor_input_dim = lagents_obs_dim
        self.agents_actor_output_dim = lagents_action_dim
        for agent_idx in range(self.agents_num):
            self.agents_critic_input_dim.append(sum(lagents_obs_dim)+sum(lagents_action_dim))
            self.agents_critic_output_dim.append(1)

        ######用list存储创建的所有agent对象#######################
        for agent_idx in range(self.agents_num):
            self.agents.append(Agent(self.agents_critic_input_dim[agent_idx],self.agents_critic_output_dim[agent_idx],self.agents_actor_input_dim[agent_idx], self.agents_actor_output_dim[agent_idx], agent_idx, alpha=alpha, beta=beta, fc1=fc1,fc2=fc2, tau=tau,chkpt_dir=chkpt_dir))
        #########################################################
    
    ## 存储所有agent的网络参数
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            agent.save_models()
    
    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()
    
    ## 所有agent选择动作，输入是个观测list
    def choose_action(self, agents_obs,explore=False):
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(agents_obs[agent_idx],explore)
            actions.append(action)
        return actions
    
    ## 学习更新网络参数（maddpg算法的核心）
    def learn(self, lstates, lactions, rewards, lnew_states, dones):
        ####################################################################
        ####将numpy转成tensor，对应list要格外注意，通常不能直接转换############
        device = self.agents[0].actor.device
        rewards = T.tensor(rewards, dtype=T.float).to(device)
        dones = T.tensor(dones).to(device)
        ####将ndarray构成的list转换成tensor list############
        lagents_new_actions = []
        lagents_actions = []
        lagents_new_states = []
        lagents_states = []

        for agent_idx, agent in enumerate(self.agents):
            new_states = T.tensor(lnew_states[agent_idx], dtype=T.float).to(device)
            lagents_new_states.append(new_states)    

            new_actions = distributions_to_onehot(agent.target_actor.forward(new_states))
            lagents_new_actions.append(new_actions)

            states = T.tensor(lstates[agent_idx],dtype=T.float).to(device)
            lagents_states.append(states)

            actions = T.tensor(lactions[agent_idx], dtype=T.float).to(device)
            lagents_actions.append(actions)
        ################################################################

        #####按列拼接成一个tensor，准备输入到网络
        new_actions = T.cat([acts for acts in lagents_new_actions], dim=1)
        actions = T.cat([acts for acts in lagents_actions],dim=1)
        new_states = T.cat([sta for sta in lagents_new_states], dim=1)
        states = T.cat([sta for sta in lagents_states], dim=1)

        ###############################################################
        ####计算损失函数，更新网络参数###################################
        ### 一次一个批量操作，actor loss就是这一批量的平均
        for agent_idx, agent in enumerate(self.agents):
            #网络的输入为tensor，返回为numpy对象
            with T.autograd.set_detect_anomaly(True):
                ######### critic网络更新############
                ## 确定critic网络目标函数
                critic_value_ = agent.target_critic.forward(new_states,new_actions).flatten()
                # print('hello,critic_value_ ',critic_value_)
                ####dones为T.tensor([True])or为T.tensor([False])，通过bool型花式切片从critic_value_选取元素，True选，False就不选
                #### 当done为true时critic_value_值为0，当done为false时保持不变。参考DQN算法
                #### 可以通过if判定 dones[:,agent_idx] == True，赋值为0
                critic_value_[dones[:,agent_idx]] = 0.0
                # print('dones[:,0]',dones[:,0])
                # print('new_critic_value_',critic_value_)
                critic_value = agent.critic.forward(states, actions).flatten()
                target = rewards[:,agent_idx] + self.gamma*critic_value_
                critic_loss = F.mse_loss(target,critic_value)
               
                ## 传播完将梯度置为零
                agent.critic.optimizer.zero_grad()
                ## 反向传播
                critic_loss.backward(retain_graph=True)
                ## critic网络参数更新
                agent.critic.optimizer.step()

                # ########## actor网络更新################
                ## actor网络目标函数
                # actor_loss = critic_value
                tem = sample_with_gumbel_softmax(actions)
                action_=(actions-tem).detach()+tem
                actor_loss = agent.critic.forward(states, action_).flatten()
                actor_loss = -T.mean(actor_loss)
                agent.actor.optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                agent.actor.optimizer.step()
                ################更新target网络参数##########################
                agent.update_network_parameters()
        ##############################################################

######## windows下必须先创建目录，才能保存参数文件############


### distribution_to_onehot函数，输入为1维np.array,返回为1维独热码
def distribution_to_onehot(distribution):
    fonehot = (distribution == distribution.max())+0.0
    if(T.sum(fonehot)!=1):
        tem0=[]
        tem1=len(fonehot)
        for i in range(tem1):
            if(fonehot[i]==1):
                tem0.append(i)
        fonehot = T.zeros(tem1)
        fonehot[np.random.choice(tem0,1)] = 1
    return fonehot

### 输入为n维，返回n维独热码，适用于批量采样
def distributions_to_onehot(distribution, eps=0.01):
    ''' 生成最优动作的独热（one-hot）形式 '''
    argmax_dis = (distribution == distribution.max(axis=1,keepdims=True)[0])+0.0
    # 生成随机动作,转换成独热形式
    rand_dis = T.eye(distribution.shape[1])
    rand_dis=rand_dis[[np.random.choice(range(distribution.shape[1]), size=distribution.shape[0])
        ]]
    # 通过epsilon-贪婪算法来选择用哪个动作
    return T.stack([
        argmax_dis[i] if r > eps else rand_dis[i]
        for i, r in enumerate(T.rand(distribution.shape[0]))
    ])


#### gumbel_softmax采样，输入为分布，返回一个分布
def sample_with_gumbel_softmax(distribution,temperature=1,eps=1e-20):
    size=distribution.shape
    noise = T.rand(size)
    distribution =  (T.log(distribution) - T.log(-T.log(noise+eps)+eps))/temperature
    tem=T.exp(distribution)
    tem = tem/tem.sum()
    return tem

#### 轮盘法采样，输入为分布，返回独热码
def sample_with_random(distribution):
    tem_len=len(distribution)
    temp0=np.random.choice(tem_len,p=distribution)###从agent的action维数中进行采样，输出为0-dim中的一个int型数值
    temp1 = T.zeros(tem_len)
    temp1[temp0] = 1
    return temp1





######## windows下必须先创建目录，才能保存参数文件############
def os_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
##########################################################

######## 将dict转成list

if __name__ == '__main__':
    ## 并行环境,render_mode = "human"时，自动开启render，max_cycles设置每个episode的迭代次数；render()可以通过close()关闭
    env = simple_adversary_v2.parallel_env(render_mode = "ansi",max_cycles=50)

    RUN_FLAG = 0
    EVALUATE = 0
    RENDER_FLAG = 0
    if RUN_FLAG:
        PRINT_INTERVAL = 500
        N_GAMES = 30000
        MAX_STEPS = 25
        LEARN_STEPS = 100
        MEAN_STEPS = 100
        BUF_SIZE = 10000
        BATCH_SIZE = 10
    else:
        PRINT_INTERVAL = 2
        N_GAMES = 2
        MAX_STEPS = 2
        LEARN_STEPS = 5
        MEAN_STEPS = 10
        BUF_SIZE = 100
        BATCH_SIZE = 2

    dobs=env.reset()
    scenario='simple_adversary_v2'
    agents_num=env.num_agents
    print(agents_num)
    print('obs',dobs)
    print('env.action_spaces',env.action_spaces)
    print('env.observation_spaces',env.observation_spaces)

    print('type of env.action_spaces',type(env.action_spaces))
    print('type of env.observation_spaces',type(env.observation_spaces))
    #####################建立保持目录####################
    #windows如果没有存储目录，会报错，可通过os.getcwd() 获取当前目录，或者用./
    #chkpt_dir = os.getcwd() + '/chkpt'+'_'+ scenario
    chkpt_dir = './chkpt'+'_'+ scenario
    os_mkdir(chkpt_dir)
    ####################网络输入维数的确定##################################
    ###lobs_dim存储每个agent观测空间的维数，laction_dim存储每个agent的action维数，都为list，便于构造buffer及确定网络的输入维数
    lobs_dim = []
    laction_dim = []

    for i in range(agents_num):
        lobs_dim.append(env.observation_space(env.agents[i]).shape[0])
        laction_dim.append(env.action_space(env.agents[i]).n)
    # print(laction_dim)

    total_steps = 0
    score_history = []
    best_score = 0   

    ########################创建maddpg对象############################
    maddpg_agents = MADDPG(lobs_dim, laction_dim, agents_num,  fc1=1, fc2=1, alpha=0.01, beta=0.01, scenario=scenario, chkpt_dir= chkpt_dir)
    ################################################################

    ###########建立缓存#############################################
    buffer = MultiAgentReplayBuffer(BUF_SIZE, lobs_dim, laction_dim, agents_num, batch_size=BATCH_SIZE)
    ###############################################################

    #####################episode迭代###########################
    for i in range(N_GAMES):
        obs = env.reset()
        score = 0
        ldone = [False]*agents_num

        ######################每个episode###########################
        while not any(ldone):
            if RENDER_FLAG:
                env.render() #关闭动画，render=‘human’自动开启
            ###########将观测值从dict转成list#################
            lobs = list(obs.values())##obs为dict，dict.values获得键值
            #########################选择actions###########################
            ####输入观测,输出独热码，
            lactions = maddpg_agents.choose_action(lobs)
            ##### 根据独热码，获取取1的位置
            lagent_actions=[]
            for i in range(agents_num):
                lagent_actions.append(lactions[i].argmax())
            #########################将action list转成dict，pettingzoo环境要求###################
            dagent_actions={agent_name:agent_action for agent_name,agent_action in zip(env.agents,lagent_actions)}  
            #####执行actions，得到奖励及下一个观测########
            obs_, rewards, terminations, truncations, infos = env.step(dagent_actions)
            ######根据terminations或truncations给出回合截止条件
            ldone = [list(terminations.values())[i] or list(truncations.values())[i] for i in range(len(ldone))]
            lobs_ = list(obs_.values())##obs为dict，dict.values获得键值
            lrewards = list(rewards.values())##rewards为dict，dict.values获得键值
            ####存储，供学习
            buffer.store_transition(lobs, lactions, lrewards, lobs_, ldone)
            ####更新观测######
            lobs = lobs_
            ####计算分数（回报）
            score += sum(lrewards)
            total_steps += 1
            ###网络的学习，更新网络参数
            if total_steps % LEARN_STEPS ==0 and not EVALUATE:
                if buffer.ready():
                    sobs, sactions, srewards, sobs_, sdone = buffer.sample_buffer()
                    maddpg_agents.learn(sobs, sactions, srewards, sobs_, sdone)
        ###存储所有步的分数（回报）###
        score_history.append(score)
        ###滑动窗口求平均分数######
        avg_score = np.mean(score_history[-MEAN_STEPS:])
       
        if not EVALUATE:
            if avg_score > best_score:
                maddpg_agents.save_checkpoint()
                best_score = avg_score
        if i % PRINT_INTERVAL == 0 and i>0:
            print('episode',i, 'average score {:.1f}'.format(avg_score))







