
# coding: utf-8

# In[1]:


from collections import deque
import os
import random
from tqdm import tqdm

import torch

from utils_drl import Agent
from utils_env import MyEnv
from utils_memory import ReplayMemory, PRMemory

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

use_dueling = True
use_PR = True
use_DDQN = True
use_MotionContinuity = True


# In[2]:


GAMMA = 0.99
GLOBAL_SEED = 0
MEM_SIZE = 100_000
RENDER = False
SAVE_PREFIX = "./models"
STACK_SIZE = 4

EPS_START = 0.5
EPS_END = 0.0
EPS_DECAY = 1000000

BATCH_SIZE = 32
POLICY_UPDATE = 4
TARGET_UPDATE = 10_000
WARM_STEPS = 50_000
MAX_STEPS = 50_000_000
EVALUATE_FREQ = 100_000

rand = random.Random()
rand.seed(GLOBAL_SEED)
new_seed = lambda: rand.randint(0, 1000_000)
if not os.path.exists(SAVE_PREFIX):
    os.mkdir(SAVE_PREFIX)

torch.manual_seed(new_seed())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = MyEnv(device)
agent = Agent(
    env.get_action_dim(),
    device,
    GAMMA,
    new_seed(),
    EPS_START,
    EPS_END,
    EPS_DECAY,
    use_dueling = use_dueling,
    use_PR = use_PR,
    use_DDQN = use_DDQN
)
if not use_PR:
    memory = ReplayMemory(STACK_SIZE + 1, MEM_SIZE, device)
else:
    memory = PRMemory(STACK_SIZE + 1, MEM_SIZE, device)


# In[ ]:


#### Training ####
obs_queue: deque = deque(maxlen=5)
done = True

progressive = tqdm(range(MAX_STEPS), total=MAX_STEPS,
                   ncols=50, leave=False, unit="b")
for step in progressive:
    if done:
        last_act = 0
        observations, _, _ = env.reset()
        for obs in observations:
            obs_queue.append(obs)

    training = len(memory) > WARM_STEPS
    state = env.make_state(obs_queue).to(device).float()
    action = agent.run(state, training)
    obs, reward, done = env.step(action)
    
    if done:
        reward -= 100
    if last_act != action:
        reward -= 0.2
        if last_act == 3-action:#相反动作
            reward -= 0.3

    last_act = action
    
    obs_queue.append(obs)
    memory.push(env.make_folded_state(obs_queue), action, reward, done)
    
    if step % POLICY_UPDATE == 0 and training:
        agent.learn(memory, BATCH_SIZE)

    if step % TARGET_UPDATE == 0:
        agent.sync()

    if step % EVALUATE_FREQ == 0:
        avg_reward, frames = env.evaluate(obs_queue, agent, render=RENDER)
        with open("rewards.txt", "a") as fp:
            fp.write(f"{step//EVALUATE_FREQ:3d} {step:8d} {avg_reward:.1f}\n")
        if RENDER:
            prefix = f"eval_{step//EVALUATE_FREQ:03d}"
            os.mkdir(prefix)
            for ind, frame in enumerate(frames):
                with open(os.path.join(prefix, f"{ind:06d}.png"), "wb") as fp:
                    frame.save(fp, format="png")
        agent.save(os.path.join(
            SAVE_PREFIX, f"model_{step//EVALUATE_FREQ:03d}"))
        done = True


# In[ ]:


batch_size=32
state_batch, action_batch, reward_batch, next_batch, done_batch, idxs, ISWeights = memory.sample(batch_size)

values = agent.__policy(state_batch.float()).gather(1, action_batch)
values_next = agent.__target(next_batch.float()).max(1).values.detach()
expected = (agent.__gamma * values_next.unsqueeze(1)) * (1. - done_batch) + reward_batch
            
abs_errors = torch.abs(expected - values).data.cpu().numpy()


# In[5]:


ISWeights


# In[21]:


memory.tree.get_min()


# In[19]:


1/MEM_SIZE/min_prob


# In[15]:


import numpy as np
n=32
pri_seg = memory.tree.total_p / n       # priority segment
memory.beta = np.min([1., memory.beta + memory.beta_increment_per_sampling])  # max = 1

min_prob = memory.tree.get_min() / memory.tree.total_p     # for later calculate ISweight
if min_prob == 0:
    min_prob = 0.00001 / memory.tree.total_p
for i in range(n):
    a, b = pri_seg * i, pri_seg * (i + 1)
    v = np.random.uniform(a, b)
    idx, p, data = memory.tree.get_leaf(v)
    prob = p / memory.tree.total_p
    ISWeights[i, 0] = np.power(prob/min_prob, -memory.beta)
    #print(b_states[i].shape, data[0][0:4].shape)
    print(1/prob/MEM_SIZE , p ,ISWeights[i, 0])


# In[13]:


max(memory.tree.tree[-MEM_SIZE:])

