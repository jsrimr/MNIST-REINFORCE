import argparse
from collections import deque
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

# Hyperparameters
# learning_rate = 0.0002
gamma = 1.0
BATCH_SIZE = 128
N_LABEL = 10

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.data = deque([])
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

        self.optimizer = optim.Adam(self.parameters())

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.softmax(x, dim=1)
        return output

    def put_data(self, item):
        self.data.append(item)

    def train_net(self):
        R = 0
        self.optimizer.zero_grad()

        loss = torch.empty([self.data[0][1].size(0) , 1])
        while self.data:
            (r, prob) = self.data.popleft()
            # R = r + gamma * R
            # loss += -torch.log(prob) * R.unsqueeze(1)
            loss += -torch.log(prob) * r.unsqueeze(1)
        loss.mean().backward()
        self.optimizer.step()



class MnistEnv:
    def __init__(self, mode='train', **kwargs):

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        if mode == 'train':
            self.dataset = datasets.MNIST('../data', train=True, download=True,
                                          transform=transform)
        elif mode == 'test':
            self.dataset = datasets.MNIST('../data', train=False,
                                          transform=transform)

        self.dataloader = iter(
            torch.utils.data.DataLoader(self.dataset, batch_size=BATCH_SIZE))
        self.state_size = self.dataset[0][0].numpy().shape
        self.action_size = N_LABEL  # TODO : remove hard code

    def reset(self):
        # self.score = 0
        self.dataloader = iter(
            torch.utils.data.DataLoader(self.dataset, batch_size=BATCH_SIZE))
        self.state, self.y = next(self.dataloader, [None, None])


        return self.state

    def step(self, action):
        reward = torch.where(self.y == action, 1, 0)
        self.state, self.y = next(self.dataloader, [None, None])

        done = True if self.state == None else False
        return (self.state, reward, done, None)


def main():
    env = MnistEnv(mode=args.mode)
    pi = Policy()
    
    print_interval = 20
    max_score = 0
    step = 0
    # for n_epi in range(10000):
    if args.mode == 'train':
        
        # for step in range(10000):
        for epoch in range(10):
            s = env.reset()
            done = False
            score = 0.0
            while not done:
                prob = pi(s)
                m = Categorical(prob)
                a = m.sample()
                s_prime, r, done, info = env.step(a)
                pi.put_data((r, prob.gather(1,a.unsqueeze(1))))
                s = s_prime
                score += (r.float().mean())

                pi.train_net()
                step += 1

                if step % print_interval == 0 and step != 0:
                    print(f"# of step :{step}, mini-batch acc : {r.float().mean()}")

            if score > max_score:
                max_score = score
                torch.save(pi.state_dict(), "checkpoint.pth")
    
    else:
        checkpoint = torch.load('checkpoint.pth')
        pi.load_state_dict(checkpoint)

        s = env.reset()
        done = False
        score = []
        while not done:
            prob = pi(s)
            m = Categorical(prob)
            a = m.sample()
            s_prime, r, done, info = env.step(a)
            s = s_prime
            score.append(r.float().mean())

        print(f"test_score : {sum(score) / len(score)}")
    # env.close()

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--mode', type=str, default='train',
                        help='train or test')
    args = parser.parse_args()

    main()
