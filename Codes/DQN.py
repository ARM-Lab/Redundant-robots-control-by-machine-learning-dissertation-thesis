import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from collections import deque
import time
from random import uniform
import random
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import math
import wandb
from wandb.keras import WandbCallback

LOAD_MODEL =None #"model_name.model"

wandb.init(project="Project1")
DISCOUNT = 0.95
REPLAY_MEMORY_SIZE = 10000  # 10 000 How many last steps to keep for model training 10000
MIN_REPLAY_MEMORY_SIZE = 1000  #1000 Minimum number of steps in a memory to start training 1000
MINIBATCH_SIZE = 100  # 100 How many steps (samples) to use for training 100
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)  before 5
MODEL_NAME = '3x12'
MIN_REWARD = -200  # For model save
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 10000  # 20_000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01

#  Stats settings
AGGREGATE_STATS_EVERY = 1 # episodes
SHOW_PREVIEW = False

class RoboEnv:
    ACTION_SPACE_SIZE = 12
    def __init__(self):
        self.state = None
        self.act = None
        self.delta1 = None
        self.delta2 = None
        self.delta3 = None
        self.reward = 0
        self.d1 = 0
        self.d2 = 0
        self.d3 = 0
        self.Xact = 0
        self.Yact = 0
        self.Zact = 110
        self.condition = []
        self.value = []
        self.episode_step = 0

    def mat(self, d1, d2, d3):
        r = 5
        lq = 110

        l1 = lq + d1
        l2 = lq + d2
        l3 = lq + d3

        if d1 == d2 and d1 == d3 or l1 == l2 and l1 == l3:

            self.Xact = 0
            self.Yact = 0
            self.Zact = lq
            #
            self.delta1 = abs(self.Xtar) - abs(self.Xact)
            self.delta2 = abs(self.Ytar) - abs(self.Yact)
            self.delta3 = abs(self.Ztar) - abs(self.Zact)

            self.delta1 = math.floor(self.delta1 * 10_000) / 10_000
            self.delta2 = math.floor(self.delta2 * 10_000) / 10_000
            self.delta3 = math.floor(self.delta3 * 10_000) / 10_000
            return self.delta1, self.delta2, self.delta3

        else:
            kappa = (2 * math.sqrt(math.pow(l1, 2) + math.pow(l2, 2) + math.pow(l3, 2) - l1 * l2 - l1 * l3 - l2 * l3)) / \
                    (r * (l1 + l2 + l3))

            theta = (2 * math.sqrt(math.pow(l1, 2) + math.pow(l2, 2) + math.pow(l3, 2) - l1 * l2 - l1 * l3 - l2 * l3)) / \
                    (r * 3)

            fi = math.atan2(math.sqrt(3) * (l2 + l3 - 2 * l1), (3 * (l2 - l3)))

            if 1 / 2 * math.pi >= theta >= -1 / 2 * math.pi and kappa > 0:
                self.Xact = (2 / kappa) * math.pow(math.sin(kappa * lq / 2), 2) * math.cos(fi)
                self.Yact = (2 / kappa) * math.pow(math.sin(kappa * lq / 2), 2) * math.sin(fi)
                self.Zact = (1 / kappa) * math.sin(kappa * lq)
                # print("ok")
                self.delta1 = abs(self.Xtar) - abs(self.Xact)
                self.delta2 = abs(self.Ytar) - abs(self.Yact)
                self.delta3 = abs(self.Ztar) - abs(self.Zact)

                self.delta1 = math.floor(self.delta1 * 10_000) / 10_000
                self.delta2 = math.floor(self.delta2 * 10_000) / 10_000
                self.delta3 = math.floor(self.delta3 * 10_000) / 10_000
                return self.delta1, self.delta2, self.delta3  # actual XYZ coordinates of the robots tip
            else:
                self.Xact = 0
                self.Yact = 0
                self.Zact = lq
                #
                self.delta1 = abs(self.Xtar) - abs(self.Xact)
                self.delta2 = abs(self.Ytar) - abs(self.Yact)
                self.delta3 = abs(self.Ztar) - abs(self.Zact)

                self.delta1 = math.floor(self.delta1 * 10_000) / 10_000
                self.delta2 = math.floor(self.delta2 * 10_000) / 10_000
                self.delta3 = math.floor(self.delta3 * 10_000) / 10_000
                return self.delta1, self.delta2, self.delta3  # out of workspace

    def mat2(self, g1, g2, g3):
        r = 5
        lq = 110

        l1 = lq + g1
        l2 = lq + g2
        l3 = lq + g3

        if g1 == g2 and g1 == g3 or g1 == g2 and g1 == g3:

            self.Xact = 0
            self.Yact = 0
            self.Zact = lq

            return self.Xact, self.Yact, self.Zact

        else:
            kappa = (2 * math.sqrt(math.pow(l1, 2) + math.pow(l2, 2) + math.pow(l3, 2) - l1 * l2 - l1 * l3 - l2 * l3)) / \
                    (r * (l1 + l2 + l3))
            theta = (2 * math.sqrt(math.pow(l1, 2) + math.pow(l2, 2) + math.pow(l3, 2) - l1 * l2 - l1 * l3 - l2 * l3)) / \
                    (r * 3)
            fi = math.atan2(math.sqrt(3) * (l2 + l3 - 2 * l1), (3 * (l2 - l3)))

            if 1 / 2 * math.pi >= theta >= -1 / 2 * math.pi and kappa > 0:
                self.Xact = (2 / kappa) * math.pow(math.sin(kappa * lq / 2), 2) * math.cos(fi)
                self.Yact = (2 / kappa) * math.pow(math.sin(kappa * lq / 2), 2) * math.sin(fi)
                self.Zact = (1 / kappa) * math.sin(kappa * lq)

                return self.Xact, self.Yact, self.Zact  # actual XYZ coordinates of the robots tip
            else:
                self.Xact = 0
                self.Yact = 0
                self.Zact = lq

                return self.Xact, self.Yact, self.Zact  # out of workspace

    def action(self, choice):#actionspace

        # Gives us 12 total movement options. (0,1,2,3)
        if choice == 0:
            act1 = +0.2
            act2 = +0.2
            act3 = -0.2
            return act1, act2, act3
        elif choice == 1:
            act1 = +0.2
            act2 = -0.2
            act3 = +0.2
            return act1, act2, act3
        elif choice == 2:
            act1 = -0.2
            act2 = +0.2
            act3 = +0.2
            return act1, act2, act3
        elif choice == 3:
            act1 = -0.2
            act2 = -0.2
            act3 = +0.2
            return act1, act2, act3
        elif choice == 4:
            act1 = -0.2
            act2 = +0.2
            act3 = -0.2
            return act1, act2, act3
        elif choice == 5:
            act1 = +0.2
            act2 = -0.2
            act3 = -0.2
            return act1, act2, act3
        if choice == 6:
            act1 = +0.1
            act2 = +0.1
            act3 = -0.1
            return act1, act2, act3
        elif choice == 7:
            act1 = +0.1
            act2 = -0.1
            act3 = +0.1
            return act1, act2, act3
        elif choice == 8:
            act1 = -0.1
            act2 = +0.1
            act3 = +0.1
            return act1, act2, act3
        elif choice == 9:
            act1 = -0.1
            act2 = -0.1
            act3 = +0.1
            return act1, act2, act3
        elif choice == 10:
            act1 = -0.1
            act2 = +0.1
            act3 = -0.1
            return act1, act2, act3
        elif choice == 11:
            act1 = +0.1
            act2 = -0.1
            act3 = -0.1
            return act1, act2, act3

    def mi(self, ):  # incomplete
        if abs(self.dist) - abs(self.dist_old) <= 0:
            return 1
        if self.dist == self.dist_old and abs(self.dist) > 0.6:
            return -1
        else:
            return -1

    def step(self, action):
        self.episode_step += 1
        self.act = self.action(action)

        self.d1 += self.act[0]
        self.d2 += self.act[1]
        self.d3 += self.act[2]

        if -20 > self.d1 > 20:
            self.d1 = 0
        if -20 > self.d2 > 20:
            self.d2 = 0
        if -20 > self.d3 > 20:
            self.d3 = 0

        # old euclidian distacne
        a = np.array((self.Xtar, self.Ytar, self.Ztar))
        c = np.array((self.Xact, self.Yact, self.Zact))
        self.dist_old = np.linalg.norm(a - c) # previous euclidian distance
       

        # new euclidian distance
        next_state = np.array([self.mat(self.d1, self.d2, self.d3)])
        b = np.array((self.Xact, self.Yact, self.Zact))
        self.dist = np.linalg.norm(a - b)  # new euclidian distance from target point to achieved point

        self.state = next_state

        distance_max = 150

        distance_reward = (self.dist / distance_max)

        reward = -distance_reward+self.mi()
        done = False
        if -0.5 <= self.dist <= 0.5:
            reward = reward + 100
            done = True
        elif self.episode_step >= 400:
            reward = reward -100
            done=True

        # self.state = next_state
        return self.state, reward, done

    def reset(self):
        self.episode_step = 0
        if len(agent.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            self.act = self.action(random.randint(0, 11))

            self.d1 += self.act[0]
            self.d2 += self.act[1]
            self.d3 += self.act[2]

            observation = np.array([self.mat(self.d1, self.d2, self.d3)])

            return observation
        else:
            self.d1 = 0
            self.d2 = 0
            self.d3 = 0

            self.delta1 = abs(self.Xtar) - 0
            self.delta2 = abs(self.Ytar) - 0
            self.delta3 = abs(self.Ztar) - 110

            observation = np.array([self.delta1, self.delta2, self.delta3])
            return observation


    def resetTar(self):
        g1 = round(uniform(-5, 5), 1)
        g2 = round(uniform(-5, 5), 1)
        g3 = round(uniform(-5, 5), 1)

        new_tar=np.array([self.mat2(g1,g2,g3)])
        self.Xtar = new_tar[0,0]
        self.Ytar = new_tar[0,1]
        self.Ztar = new_tar[0,2]
        return new_tar



    def render(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        labels = ["Initioal Position", "Achieved position", "Desired position"]

        ax.scatter(self.Xact, self.Yact, self.Zact, marker="o", label=labels[1])
        ax.scatter(self.Xtar, self.Ytar, self.Ztar, marker="x", label=labels[2])
        ax.scatter(0, 0, 110, marker="x", label=labels[0])

        ax.set_xlabel('x-axis')
        ax.set_ylabel('y-axis')
        ax.set_zlabel('z-axis')

        ax.legend(loc="best")

        plt.show()
        #plt.pause(2)
       # plt.close()


env = RoboEnv()

ep_rewards = [0]
epi_dist = []
MaximalQ = []
ep_steps = []

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.compat.v1.set_random_seed(1)

# Create models folder
if not os.path.isdir('models3'):
    os.makedirs('models3')
# ============================================================================================


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = self.log_dir
        self._log_write_dir = os.path.join(self.log_dir, MODEL_NAME)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        self.model = model

        self._train_dir = os.path.join(self._log_write_dir, 'train')
        self._train_step = self.model._train_counter

        self._val_dir = os.path.join(self._log_write_dir, 'validation')
        self._val_step = self.model._test_counter
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)
        with self.writer.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key, value, step=self.step)
                self.writer.flush()
        # self._write_logs(stats, self.step)

    # Added because of version
    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()
#============================================================================================

class DQNAgent:
    def __init__(self):

        # main model - gets trained every step
        self.model = self.create_model()

        # Target model- this is what we .predict against every step, not trained but updated from the main model
        # (stability of the system)
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)  # list that has max size,

        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")

        self.target_update_counter = 0


    def create_model(self):

        if LOAD_MODEL is not None:
            print(f"Loading {LOAD_MODEL}")
            model = load_model(LOAD_MODEL)
            print(f"Model {LOAD_MODEL} loaded!")
        else:
            model = Sequential()

            model.add(Dense(340, input_shape=(3,), activation='softmax',
                            name="layer"))  # input layer +1st hidden...input_shape

            model.add(Dense(340, activation='relu', name="layer2"))  # 3 hiden layers

            model.add(Dense(340, activation='relu', name="layer3"))

            model.add(Dense(12, activation='softmax'))  # output layer
            model.compile(loss="mse", optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self, terminal_state, step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE) #sample size of MINIBATCH_SIZE
        current_states = np.vstack([transition[0] for transition in minibatch]).astype(object) # transition in environment, transition[0] = current state

        scaler = MinMaxScaler(feature_range=(0, 1))
        current_states = scaler.fit_transform(current_states.reshape(-1, 3))
        current_states = current_states.tolist()

        current_qs_list = self.model.predict(current_states) #current_states

        new_current_states = np.vstack([transition[3] for transition in minibatch]).astype(object)  # transition[3] = new_state

        scaler = MinMaxScaler(feature_range=(0, 1))
        new_current_states = scaler.fit_transform(new_current_states.reshape(-1, 3))

        new_current_states = new_current_states.tolist()
        future_qs_list = self.target_model.predict(new_current_states)

        X = []  # feature sets() 
        y = []  # targets(action that were taken) 

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):  # tuple, index == 0,1,2,3
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q # target y
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)  # inputs
            y.append(current_qs)  # outputs

        self.model.fit(scaler.fit_transform(np.vstack(X)), np.vstack(y), batch_size=MINIBATCH_SIZE,
                       verbose=0, shuffle=False, callbacks=[WandbCallback()] if terminal_state else None) 


        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:  # when to update the target model
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        scaler = MinMaxScaler(feature_range=(0, 1))
        return self.model.predict(scaler.fit_transform(np.vstack(state).reshape(-1, 3)))[0] # current state, np.array(env.state).reshape(-1, *env.state.shape)

agent = DQNAgent()

for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit="episodes"):
    agent.tensorboard.step = episode

    episode_reward = 0
    step = 1
    new_target = env.resetTar()
    current_state = env.reset()  # state
    vzd = 0
    MaxQ = 0
    done = False  # reset flaq and starts iterating until episode ends

    while not done:
        if np.random.random() > epsilon:
            # Get action from Q table
            Qprediction = agent.get_qs(current_state)
            MaxQ = np.max(Qprediction)
            action = np.argmax(agent.get_qs(Qprediction))
        else:
            action = np.random.randint(0, env.ACTION_SPACE_SIZE)  # else random action is taken


        new_state, reward, done = env.step(action)

        episode_reward += reward 
        vzd += env.dist

        if SHOW_PREVIEW and len(agent.replay_memory) > MIN_REPLAY_MEMORY_SIZE and done and not episode % AGGREGATE_STATS_EVERY:
            env.render()

        agent.update_replay_memory((current_state, action, reward, new_state, done)) 
        agent.train(done, step)
        current_state = new_state
        step += 1


    # Append episode reward to a list and log stats (every given number of episodes)
    ep_steps.append(step)
    ep_rewards.append(episode_reward)
    epi_dist.append(env.dist)
    MaximalQ.append(MaxQ)


    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        err = epi_dist[len(epi_dist)-1]
        Qval = MaximalQ[len(MaximalQ) - 1]
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward,
                                      epsilon=epsilon)
        wandb.log({'average_reward': average_reward, 'episode': episode})
        wandb.log({'min_reward': min_reward, 'episode': episode})
        wandb.log({'max_reward': max_reward, 'episode': episode})
        wandb.log({'Error': err, 'episode': episode})
        wandb.log({'Qvalue': Qval, 'episode': episode})
        wandb.log({'Steps': ep_steps, 'episode': episode})

        # Save model, but only when min reward is greater or equal a set value
        if err <= 30: 
            agent.model.save(
                f'models3/{MODEL_NAME}__{err:_>7.2f}Error_{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
