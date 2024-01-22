# script contains the various optimizers, and their supporting explaination

from torch import optim
from torch import nn
import torch.nn.functional as F
from torch.optim import lr_scheduler as lr

model = nn.Linear()

# basic optimizer example
opt1 = optim.SGD(model.parameters(), 
                 lr=0.01,
                 momentum=0.9,
                 weight_decay=0.2,
                 dampening=1.5)

opt2 = optim.Adam(model.parameters(),
                  lr=0.01)
loss_fn = nn.MSELoss()

# back propagation loop
for ind, tar in dataset:
    # make the grads to 0 
    opt1.zero_grad()
    # make a pred
    pred = model(ind)
    # get loss
    loss = loss_fn(pred, tar) 
    # loss backward
    loss.backward()
    # update the model param after optimising them
    opt1.step()

for input, target in dataset:
    # Use the closure in special optimizer where the functions are revaluated
    def closure():
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        return loss
    optimizer.step(closure)


# Following Optimizers are available
"""
Adadelta : Implements Adadelta algorithm.

Adagrad: Implements Adagrad algorithm.

Adam: Implements Adam algorithm.

AdamW: Implements AdamW algorithm.

SparseAdam: SparseAdam implements a masked version of the Adam algorithm suitable for sparse gradients.

Adamax: Implements Adamax algorithm (a variant of Adam based on infinity norm).

ASGD: Implements Averaged Stochastic Gradient Descent.

LBFGS: Implements L-BFGS algorithm, heavily inspired by minFunc.

NAdam: Implements NAdam algorithm.

RAdam: Implements RAdam algorithm.

RMSprop: Implements RMSprop algorithm.

Rprop: Implements the resilient backpropagation algorithm.

SGD: Implements stochastic gradient descent (optionally with momentum).
"""

# Pytorch has 3 major categories of implementations: for-loop, foreach (multi-tensor), and fused. 

# Scheduler comes after the optimizer both in initialisation and step()

scheduler = lr.ExponentialLR(optimizer=opt1, gamma=0.9)
scheduler2 = lr.MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
# Scheduling is done every epoch
for ep in range(epochs):
    for ind, tar in dataset:
        opt1.zero_grad()
        # forward
        pred = model(ind)
        loss = loss_fn(pred, tar)
        # backward
        loss.backward
        opt1.step()
    scheduler.step()
    scheduler2.step()
"""
lr_scheduler.LambdaLR: Sets the learning rate of each parameter group to the initial lr times a given function.

lr_scheduler.MultiplicativeLR: Multiply the learning rate of each parameter group by the factor given in the specified function.

lr_scheduler.StepLR: Decays the learning rate of each parameter group by gamma every step_size epochs.

lr_scheduler.MultiStepLR: Decays the learning rate of each parameter group by gamma once the number of epoch reaches one of the milestones.

lr_scheduler.ConstantLR: Decays the learning rate of each parameter group by a small constant factor until the number of epoch reaches a pre-defined milestone: total_iters.

lr_scheduler.LinearLR: Decays the learning rate of each parameter group by linearly changing small multiplicative factor until the number of epoch reaches a pre-defined milestone: total_iters.

lr_scheduler.ExponentialLR: Decays the learning rate of each parameter group by gamma every epoch.

lr_scheduler.PolynomialLR: Decays the learning rate of each parameter group using a polynomial function in the given total_iters.

lr_scheduler.CosineAnnealingLR: Set the learning rate of each parameter group using a cosine annealing schedule, where nmax is set initial lr, TCur is the number of epochs in last restart. 

lr_scheduler.ChainedScheduler: Chains list of learning rate schedulers.

lr_scheduler.SequentialLR: Receives the list of schedulers that is expected to be called sequentially during optimization process and milestone points that provides exact intervals to reflect which scheduler is supposed to be called at a given epoch.

lr_scheduler.ReduceLROnPlateau: Reduce learning rate when a metric has stopped improving.

lr_scheduler.CyclicLR: Sets the learning rate of each parameter group according to cyclical learning rate policy (CLR).

lr_scheduler.OneCycleLR: Sets the learning rate of each parameter group according to the 1cycle learning rate policy.

lr_scheduler.CosineAnnealingWarmRestarts: Set the learning rate of each parameter group using a cosine annealing schedule, where 
"""