###### Tue May 17 21:02:24 CST 2022
<font color=red>Ctrl+F5插入以上时间</font>
markdown文件插入索引（代码方法参考markdpad2，vscode中markdown toc插件，右击选项插入索引，插入带标号的索引，前提标题要规范，最后\#后面加空格）

![p](图片/xxx.jpg)

###### Mon Apr 3 14:55:44 CST 2023
## 工程p-maddpg-project



### gym环境问题

#### 动作空间与状态空间
每种环境的动作空间、观测空间的真实物理意义要通过源码才能知道
比如CartPole类（class CartPoleEnv(gym.Env)）可以通过源码知道:
https://github.com/openai/gym/blob/0cd9266d986d470ed9c0dd87a41cd680b65cfe1c/gym/envs/classic_control/cartpole.py
动作空间是一个离散数据:   状态空间值{0,1},0--表示左移动，1--表示右移动
状态空间是一个多维空间，四个维度分别表示：小车在轨道上的位置，杆子和竖直方向的夹角，小车速度，角度变化率。



###### ddpg梯度的实现
<font color=red>
https://hrl.boyuai.com/chapter/3/%E5%A4%9A%E6%99%BA%E8%83%BD%E4%BD%93%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E8%BF%9B%E9%98%B6/
</font>

###### logits
深度学习中logits就是最终的全连接层的输出，而非其本意

##### Box类
Box类对应于多维连续空间
Box空间可以定义多维空间，每一个维度可以用一个最低值和最大值来约束
定义一个多维的Box空间需要知道每一个维度的最小最大值，当然也要知道维数

Box类可以通过.shape查看维数
print('observation space',env.observation_space[0].shape) 第0个智能体观测空间

##### Discrete类
Discrete类对应于一维离散空间
定义一个Discrete类的空间只需要一个参数n就可以了
discrete space允许固定范围的非负数

Discrete类可以通过.n函数查看维数
print('action space',env.action_space[0].n)
查看第0个智能体动作空间维数


#### OpenAI常见指令
1. env.reset() 初始化环境
2. env.step() 在当前状态执行指定的动作
3. env.render() 显示（渲染）当前环境
4. env.close() 关闭环境
5. env.seed(number)#number为固定值，print(env.reset()),初始化为固定值,不设置，初始化为随机值



### Gymnasium
gymnasium是gym的升级版，对gym的API更新了一波，也同时重构了一下代码。利用Gymnasium自定义自己的环境

https://zhuanlan.zhihu.com/p/621036937

https://gymnasium.farama.org/api/spaces/
####  gymnasium.spaces 各种空间
import gymnasium.spaces as spaces

##### Discrete类
observation_space = Discrete(2, seed=42) # {0, 1}
...observation_space.sample()
...输出0
observation_space = Discrete(3, start=-1, seed=42)  # {-1, 0, 1}

#### wrapper
wrapper就是在Env外面再包一层，不用去修改底层Env的代码就可以改变一个现有的环境。可以修改step返回的信息，action传入的信息，等等。
其实就是充当agent与Env之间的一个中间层。
一共有四类wrapper：
1. Misc wrapper：
2. Action
3. Observation
4. Reward



### pettingzoo环境的代码

#### 环境配置
首先打开Anaconda Prompt
1. conda create --name pettingzoo-test python==3.10
2. activate pettingzoo-test
3. pip install pip install pettingzoo[mpe] -i https://pypi.tuna.tsinghua.edu.cn/simple
4. pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple
6. pip install ipykernel -i https://pypi.tuna.tsinghua.edu.cn/simple

#### pettingzoo使用说明
中文说明
https://zhuanlan.zhihu.com/p/615930331
https://zhuanlan.zhihu.com/p/616167697
https://zhuanlan.zhihu.com/p/616210050
英文说明
https://pettingzoo.farama.org/content/basic_usage/

#### pettingzoo下面的MPE环境
主文件通过from pettingzoo.mpe import simple_adversary_v2来实现环境的调用
<font color=red> 以Simple Adversary环境为例，SimpleEnv类很重要</font> 
调用
env = simple_adversary_v2.parallel_env()
调用simple_adversary文件里面的方法
...
<font color=red> class raw_env(SimpleEnv, EzPickle):这个类很重要，里面有 world = scenario.make_world(N)，并继承SimpleEnv，这个环境下面有step等基本操作  
...
env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)
</font> 
其中parallel_wrapper_fn方法里面有如下的调用
env = aec_to_parallel_wrapper(env)
上述对象由下面类构造
class aec_to_parallel_wrapper(ParallelEnv):

class AECEnv:

##### 观测空间、动作空间、奖励空间
<font color=red>pettingzoo下面的mpe都是用dict来存储，不同于mpe</font>

#### step()中action的问题
<font color=red>pettingzoo中step输入为dict型action，不同于mpe环境为array构成的list。</font>例如simple_adversary环境中
action={'adversary_0': Discrete(5), 'agent_0': Discrete(5), 'agent_1': Discrete(5)}
simple_adversary的action空间为
Agent action space: [no_action, move_left, move_right, move_down, move_up]
Adversary action space: [no_action, move_left, move_right, move_down, move_up]
对应[0,1,2,3,4]
举例，step({'adversary_0': 1, 'agent_0': 2, 'agent_1': 3})


#### render()渲染
render依赖于模式，当render模式为human时，自动运行env.render()

#### 回合终止的判定 terminations, truncations
truncations是一个dict，键为智能体名字，值为bool型，值由迭代次数来确定，当超过迭代次数是为Ture，其它为False。比如class raw_env(SimpleEnv, EzPickle)初始化对象时，max_cycles=25
如何超过迭代次数，truncations
... terminations, truncations, infos = env.step(action)
env.agents中的agents会消亡

##### 通过agent来判定终止条件
意思是可以用not env.agents来判断游戏是否结束

实操时，“当智能体被终止或截断时，它将从agents中移除”这个功能需要用self._was_dead_step(action)函数。

#### 学习
会涉及动作空间离散--求导的问题，需要额外注意。


### maddpg代码整体框架

参考pettingzoo_maddpg_class及maddpg_pettingzoo_main for pettingzoo[mpe]环境
或 mpe_maddpg_class及maddpg_mpe_main for mpe环境

#### 代码的区别表现在
class文件中agent.choose action 及maddpg.learn有些差别，main主程序有些差别，主要是list与dict的转换

#### 命名规范
l开头的变量表示list
n开头的变量表示numpy.ndarray


#### maddpg原理
https://zhuanlan.zhihu.com/p/92466991
#### 基本类介绍
class MultiAgentReplayBuffer；存储多智能体历史状态及动作数据，用于采样
class CriticNetwork：构建critic网络
class ActorNetwork：构建actor网络
class Agent：智能体模型（动作空间、状态空间）
class MADDPG：多智能体模型
__main__main函数：主程序

#### Class: MultiAgentReplayBuffer
涉及上一刻状态（观测）、动作、奖励、下一刻状态（观测）、是否结束标志。$\{ϕ(S_j),A_j,R_j,ϕ(S'_j),is\_end_j\}$及done等参数的存储。
搞清楚维数每一类的维数
##### 基本参数的初始化
结合maddpg网络架构及网络的输入来确定基本参数：智能体个数agents_num、、每个智能体观测/状态lagents_obs_dim及action的维数lagents_action_dim、批量采样buf大小（行数）batch_size、运行终止符号位的存储。
def __init__(self,buf_size,lagents_obs_dim,lagents_action_dim,agents_num,batch_size):
1. buf_size： buffer大小（行数）
2. lagents_obs_dim： 所有智能体观测/状态维数，是个list
3. lagents_action_dim：所有智能体action维数，是个list
4. agents_num:所有智能体个数，是个int
5. batch_size：采样行数，是个int

##### 基本功能
###### buffer的初始化
init_replay_buf(self):
###### 往存储空间存数据store_transition()
<font color=red>注意是基于numpy的数据类型来实现的，在调用时，要注意数据类型的变换保证正确。</font>

store_transition(self,agents_obs,action,reward,agents_new_obs,done):函数
函数的输入：
1. agents_obs存的是所有agent当前观测，是个list，每个元素代表一个agent的观测值是个numpy.ndarray，```[array([1,2,...]),array([1,2,...]), ...]```。
2. action存的是所有agent的动作，是个list，和raw_obs格式一致。
3. reword存的是所有agent的奖励，是个list，里面元素是数字，```[1,2,...]```。
4. agents_new_obs存的是所有agent执行动作后的下一次观测，是个list，和agents_obs一样。
5. done是一个回合episode的结束标志，是个二值list。```[False,False,...]```

list生成通过定义test=[],然后通过test.append()函数来实现。

所有的buffer（类别为numpy.ndarray）：

###### 采样sample_buffer():
sample_buffer(self):函数。从buf_size生成batch_size个随机整数，通过索引从buffer中选取对应行来实现采样。
生成随机整数使用
max_mem = min(self.buf_cntr,self.buf_size)
np.random.choice(max_mem, self.batch_size, replace=False)来实现；
函数的输入：self
返回采样后的buffer，长度为batch：
1. lagents_states;
2. lagents_actions;
3. rewards;
4. lagents_new_states;
5. done 一个episode结束标志。

###### ready()
保证buf_cntr大于等于batch_size，保证np.random.choice使用正确

#### Class：CriticNetwork
此处用linear网络实现nn.Linear(input_dims, fc1_dims)，对于linear网络输入输出都是二维tersor，输入[batch_size, in_features]中的in_features是输入神经元个数，输出为[batch_size, out_features]，输出神经元个数为 out_features

batch_size为批量处理的规模，配置网络时不用管。
##### 初始化

网络结构：
1. critic网络的输入维数input_dim
2. 中间层的输入维数fc1_dim
3. 中间层的输出维数fc2_dim
4. 整个网络的输出维数output_dim

优化器的选择：
1. 配置学习率beta
2. 选择优化函数

硬件的选择：
cuda or cpu

网络保持位置的设置
self.chkpt_file = os.path.join(chkpt_dir, name)

##### 基本功能
###### 前向计算forward()
输入层与中间层为relu激活
输出层不用激活函数。
###### 网络存储 save_checkpoint(self):
T.save(self.state_dict(), self.chkpt_file)
###### 网络加载 load_chckpoint(self):
self.load_state_dict(T.load(self.chkpt_file))


#### Class：ActorNetwork

##### 初始化

网络结构：
1. actor网络的输入维数input_dim
2. 中间层的输入维数fc1_dim
3. 中间层的输出维数fc2_dim
4. 整个网络的输出维数output_dim

优化器的选择：
1. 配置学习率beta
2. 选择优化函数

硬件的选择：
cuda or cpu

网络保持位置的设置
self.chkpt_file = os.path.join(chkpt_dir, name)

##### 基本功能
###### 前向计算forward()
输入层与中间层为relu激活
输出层为softmax
###### 网络存储 save_checkpoint(self):
T.save(self.state_dict(), self.chkpt_file)
###### 网络加载 load_chckpoint(self):
self.load_state_dict(T.load(self.chkpt_file))


#### Class：Agent

##### 初始化 __init__(self, critic_input_dim, critic_output_dim, actor_input_dim,  actor_output_dim, agent_idx, alpha, beta, fc1, fc2, tau, chkpt_dir):
1. agent名字通过字符串格式化命名；
2. 每个agent有四个网络：actor网络、目标actor网络、critic网络、目标critic网络
3. 网络参数的初始化（alpha、beta等）
##### 基本功能
###### 更新目标网络的参数update_network_parameters(self, tau=None)

注意用dict()函数将网络参数生成字典
1. 通过net.named_parameters()获取构建的网络的参数 actor_params；
    例如，actor_params = self.actor.named_parameters()
2. 用actor_state_dict=dict()函数将网络参数生成字典，每层网络有名字和参数构成；
    例如，actor_state_dict = dict(actor_params)
3. 通过遍历字典的所有名字，完成字典值的更新
    例如， for name in actor_state_dict:
             actor_state_dict[name] = tau*actor_state_dict[name].clone() + (1-tau)*target_actor_state_dict[name].clone()
4. 用更新后的字典，通过net.load_state_dict(actor_state_dict)
    例如，target_actor.load_state_dict(actor_state_dict)

###### 输入观测，输出动作choose_action(self, observation)
实际就是actor网络输出，只不过要进行类型转换。
pytorch网络的输入必须为tensor类型，输出也是tensor类型。由于主程序数据处理都是numpy类型，因此输入及输出需要进行转换，将numpy输入转为tensor，将tensor输出转为numpy。


1. 函数的输入为状态/观测
2. 输出动作
3. 返回numpy类型的输出，转为1维数组输出
action.detach().cpu().numpy().numpy()[0]

###### save_models(self):
保存4个网络
###### load_models(self):
加载4个网络

#### Class：MADDPG
##### 初始化__init__(self, lagents_obs_dim, lagents_action_dim, agents_num, scenario='simple', alpha=0.01, beta=0.01, fc1=64, fc2=64, gamma=0.99, tau=0.01, chkpt_dir='/chkpt'):
每个agent都要初始化，通过循环构造agent对象list来实现
agents列表存储所有agent对象
两种方法，第一种用range
for agent_idx in range(self.agents_num):
    action = self.agents[agent_idx].choose_action(raw_obs[agent_idx])
或者用enumerate（将list加上序号）
for agent_idx, agent in enumerate(self.agents):
    action = agent.choose_action(raw_obs[agent_idx])

##### 基本功能
##### 保存和加载模型
保存每个agent的参数，通过循环来实现保存或加载
##### 动作选择 choose_action(self, raw_obs): raw_obs为list
action=[]
循环产生list
action = agent.choose_action(agents_obs[agent_idx])
action += np.random.random(size=action.shape)加噪声
actions.append(action)
把所有agent的动作输出构成actions list返回

##### 网络的学习learn(self, buffer):重中之重
网络学习是基于pytorch，因此要进行格式转换。采样$m$个样本$\{ϕ(S_j),A_j,R_j,ϕ(S'_j),is\_end_j\}$等参数的存储。

1. 判定采样操作是否OK 
2. 采样lstates, lactions, rewards, lnew_states, dones = buffer.sample_buffer()
3. 确定硬件类型，然后将numpy类型的数据转成tensor类型
    1）非list直接转换成tensor：rewards = T.tensor(rewards, dtype=T.float).to(device)
    2）list通过循环转成tensor：lagents_states = []
4. 通过T.cat把所有agent的输入扩展成1维数组，就是列进行扩展。扩展成critic网络的输入维数大小
5. 网络参数学习与更新
    1）更新critic网络：计算损失函数，通过TD误差（新状态输入target的critic网络，原状态输入critic网络）来求均方误差F.mse_loss()
    2）.optimizer.zero_grad()
    3）.backward()计算梯度
    4）.optimizer.step()更新参数
    5）更新actor网络：计算损失函数，Q(s,a)关于s的期望作为目标函数，输入critic网络得到
    6）更新target网络
#### 孤立于类的其它函数

##### os_mkdir(path):
创建目录,windows下必须先创建目录，才能保存文件参数

#### 主函数__name__ == '__main__':
##### 环境基本配置
1. scenario = '':选择环境
2. env = make_env(scenario)

##### 确定维数
1. agent的数目
2. 每个agent的action用几维表示，obs用几维表示，state用几维表示
agents_obs存的是所有agent当前观测，是个list，每个元素代表一个agent的观测值是个numpy.ndarray，
```[array([1,2,...]),array([1,2,...]), ...]```。
2. action存的是所有agent的动作，是个list，和raw_obs格式一致。
3. reword存的是所有agent的奖励，是个list，里面元素是数字
3. 确定网络输入的维数critic网络，actor网络
##### 设置目录
chkpt_dir = './chkpt' + '_' + scenario
./表示当前目录，也可以用os.getcwd()获取windows系统下当前目录
os_mkdir(chkpt_dir)

##### 参数设置
1. episode数目
2. 每个episode长度
3. 学习步（多少步更新学习参数）
4. 平均步（输出reward的滑动平均步数）
5. 打印输出步（多少步，打印输出一次）
##### 建立maddpg对象
maddpg_agents = MADDPG()
##### 建立buffer对象
memory = MultiAgentReplayBuffer()
##### maddpg伪代码的实现
1. 选择动作
2. 执行
3. 存储
4. 学习
5. 统计score






#### 梯度问题


##### 梯度估计
https://kexue.fm/archives/6705/comment-page-1
在VAE或者强化学习中，我们经常会需要优化一个与采样有关的目标函数:
$${\mathcal{F}}(\theta):=\int p(\mathbf{x};\theta)f(\mathbf{x};\phi)d\mathbf{x}=\mathbb{E}_{p(\mathbf{x};\theta)}\left[f(\mathbf{x};\phi)\right]$$
其关于$\theta$的梯度：
$$\eta:=\nabla_{\theta}{\mathcal{F}}(\theta)=\nabla_{\theta}\mathbb{E}_{p(\mathbf{x};\theta)}\left[f(\mathbf{x};\phi)\right].$$
最直观的方法是直接精确的求积分再求导，得到显示的表达式再求导，但这通常不可能。所以一般使用采样来代替积分。但是如果我们直接根据$p(x,\theta)$进行采样，则关于$\theta$的梯度信息将会丢失，从而无法更新$\theta$。而梯度估计方法就是用来在这种情况下估计对于$\theta$的梯度的。

两种方法：
1. Derivatives of Measure ：对 $p(x,\theta)$（measure）进行求导来推出；
2. Derivative of Paths：根据 $f(x)$（cost function）来求导.


##### Score Function(SF) Gradient Estimator
SFGE是一个经典的Derivatives of Measure类型的方法，他的另一个名字就是 REINFORCE。Score Function 通常指对数概率分布关于分布参数$\theta$的导数$\nabla_{\theta}\log p(\mathbf{x};\theta)$，展开得到
$$\nabla_{\theta}\log p(\mathbf{x};\theta)={\frac{\nabla_{\theta}p(\mathbf{x};\theta)}{p(\mathbf{x};\theta)}}.$$
原先需要求的梯度就可以转化为：
$$\begin{align}\begin{split}
\eta &{\rm{ = }}{\nabla _\theta }{_{p({\bf{x}};\theta )}}\left[ {f({\bf{x}})} \right] = {\nabla _\theta }\int {p({\bf{x}};\theta )f({\bf{x}})d{\bf{x}}} \\
 &= \int {{\nabla _\theta }p({\bf{x}};\theta )f({\bf{x}})d{\bf{x}}} \\
 &= \int {p({\bf{x}};\theta )f({\bf{x}}){\nabla _\theta }\log (p({\bf{x}};\theta ))d{\bf{x}}} \\
 &= {_{p({\bf{x}};\theta )}}\left[ {f({\bf{x}}){\nabla _\theta }\log (p({\bf{x}};\theta ))} \right]
\end{split}
\end{align}$$
再根据蒙特卡洛采样，我们就可以得到估计的梯度：
$$\bar{\eta}_{N}=\frac{1}{N}\sum_{n=1}^{N}f(\hat{\bf x}^{(n)})\nabla_{\theta}\log p(\hat{\bf x}^{(n)};\theta);\quad\hat{\bf x}^{(n)}\sim p({\bf x};\theta).$$

###### 注意事项
对于一个估计器，首先我们需要考虑到的就是它的偏差和方差。SFGE在满足求导积分可换的条件下，是一个无偏的估计。但是其方差设法保证不能太大。
在使用SFGE时需要考虑的：

1. 任意cost function都可用。
2. 分布必须关于$\theta$可导。
3. 必须容易从该分布中采样。
4. 由于方差受很多因素影响，因此需要尝试减少方差的一些方法。

##### Pathwise Gradient Estimator
https://zhuanlan.zhihu.com/p/104991140
GE是一种Derivative of Paths的方法，通过将$\theta$引入到cost function$f(x)$ 上去来实现保留参数。其主要依靠于 Law of the Unconscious Statistician（LOTUS）:
$$\mathbb{R}_{p({\bf x};\theta)}\left[f({\bf x})\right]\underline{{{-\mathbb{E}_{p(\epsilon)}}}}\left[f(g(\epsilon;\theta))\right]$$
假设$p({\bf x};\theta)$可由base distribution $p(\epsilon)$和可导且可逆的转换函数$g(\epsilon;\theta))$得到，则原始的问题变成：

$$\begin{align}\begin{split}
\eta&=\nabla_{\theta}\mathbb{E}_{p(\mathbf{x};\theta)}\left[f(\mathbf{x})\right]=\nabla_{\theta}\int p(\mathbf{x};\theta)f(\mathbf{x})d\mathbf{x}\\ 
&=\nabla_{\theta}\int p(\epsilon)f(g(\epsilon;\theta))d\epsilon
\end{split}
\end{align}$$

###### 注意事项

在使用PGE时通常需要考虑：
1. cost function 需要可导（关于$\theta$）。
2. 如果我们可以从base distribution方便的采样并获得对应的数据点，我们只需要获得对应的转换函数而不需要知道原始的分布。
3. 如果我们的数据是从原始分布中采样出来的，我们需要知道原始分布与其转换函数的逆。
4. 我们需要通过控制cost function的连续来控制方差。

#### 动作空间离散--采样引起的求导问题
##### 理解采样为啥会影响求导

https://kexue.fm/archives/6705/comment-page-1
<font color=red>以BP神经网络为例，计算梯度时可以得到精确的表达式，要求出具体的数值则需要当前参数$w(k)$的值，$k$表示迭代，参数$w(k)$输入网络的输出值$\hat y$，目标$y$等。如果经过采样，实际上将$\hat y$替换为另一个值$\bar y$，中间有一个不可导的过程，引起梯度消失。但是按说依然可以把$\bar y$带入求导后的表达式，得到导数（手工求导写出具体表达式，将$\hat y$替换为$\bar y$是完全可以求出一个导数的），只是导数不准了？？？</font>


https://zhuanlan.zhihu.com/p/551255387

假设网络输出的三维向量代表三个动作（前进、停留、后退）在下一步的收益
vue=[-10,10,15]，那么下一步我们就会选择收益最大的动作（后退）继续执行，于是输出动作
[0,0,1]。选择值最大的作为输出动作，这样做本身没问题，但是在网络中这种取法有个问题是不能计
算梯度，也就不能更新网络
https://blog.csdn.net/weixin_40255337/article/details/83303702



##### 重参数化
采样重参数化解决梯度问题
https://kexue.fm/archives/6705/comment-page-1

##### 基于softmax的采样
1. 通过sofmax后，
2. 再依概率采样，比如直接用np.random.choice函数依照概率生成样本值，经典的采样方法就是用softmax函数加上轮盘赌方法(np.random.choice)

但这样还是会有个问题，这种方式怎么计算梯度？不能计算梯度怎么更新网络？

##### 基于gumbel-max的采样
$x=\mathop {\arg \max }\limits_i (log(a_i)+G_i)$

1. 对于网络输出的一个K维向量v（softmax后的值$v_i=log(a_i)$）,生成K个服从均匀分布U(0,1)的独立样本e1,...,ek;
2. 通过Gi=-log(-log(ei)计算得到Gi;
3. 对应相加得到新的值向量V=[v1+G1, v2+G2,..., vk+Gk]
4. 取最大值作为最终的类别

由于这中间有一个argmax操作，这是不可导依旧没法用于计算网络梯度。
##### 基于gumbel-softmax的采样
$x=softmax((log(a_i)+G_i)/temperature)$
1. 对于网络输出的一个K维向量v（softmax后的值$v_i=log(a_i)$）,生成K个服从均匀分布U(0,1)的独立样本e1,...,ek;
2. 通过Gi=-log(-log(ei)计算得到Gi;
3. 对应相加得到新的值向量V=[v1+G1, v2+G2,..., vk+Gk]
4. 通过softmax函数计算概率函数值

##### 代码实现
https://blog.csdn.net/weixin_40255337/article/details/83303702
https://zhuanlan.zhihu.com/p/551255387
<font color=red>
利用.detach() 将gumbel-max操作从计算图分离开，再在计算图上增加gumbel-softmax操作，但实际的结果还是gumbel-max
a = (a_gumbel_max - a_gumbel_softmax).detach() + a_gumbel_softmax
</font>
#### gym中的discrete类、box类和multidiscrete类

Box 连续空间->DiagGaussianPdType （对角高斯概率分布）
Discrete离散空间->SoftCategoricalPdType（软分类概率分布）
MultiDiscrete连续空间->SoftMultiCategoricalPdType （多变量软分类概率分布）
多二值变量连续空间->BernoulliPdType （伯努利概率分布


