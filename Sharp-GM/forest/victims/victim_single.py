"""Single model default victim class."""

import torch
import numpy as np
from collections import defaultdict


from ..utils import set_random_seed
from ..consts import BENCHMARK
torch.backends.cudnn.benchmark = BENCHMARK

from .victim_base import _VictimBase

import copy


def list_plus(list1, list2):# two list must have same structure
    list_total = copy.deepcopy(list1)
    for i in range(len(list_total)):
        list_total[i] += list2[i]
    return list_total
    
def list_multiply(list1, a):
    list_total = copy.deepcopy(list1)
    for i in range (len(list_total)):
        list_total[i] *= a
    return list_total

def add_gaussian(model, sigma):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        for _, param in model.named_parameters():
            param_size = param.size()
            mean_param = torch.zeros(param_size, device=device)
            std_param = sigma * torch.ones(param_size, device=device)
            gaussian_noise = torch.normal(mean_param, std_param)
            param.add_(gaussian_noise)


class _VictimSingle(_VictimBase):
    """Implement model-specific code and behavior for a single model on a single GPU.

    This is the simplest victim implementation.

    """

    """ Methods to initialize a model."""

    def initialize(self, seed=None):
        if self.args.modelkey is None:
            if seed is None:
                self.model_init_seed = np.random.randint(0, 2**32 - 1)
            else:
                self.model_init_seed = seed
        else:
            self.model_init_seed = self.args.modelkey
        set_random_seed(self.model_init_seed)
        self.model, self.defs, self.criterion, self.optimizer, self.scheduler = self._initialize_model(self.args.net[0])

        self.model.to(**self.setup)
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        print(f'{self.args.net[0]} model initialized with random key {self.model_init_seed}.')

    """ METHODS FOR (CLEAN) TRAINING AND TESTING OF BREWED POISONS"""

    def _iterate(self, kettle, poison_delta, max_epoch=None):
        """Validate a given poison by training the model and checking target accuracy."""
        stats = defaultdict(list)

        if max_epoch is None:
            max_epoch = self.defs.epochs

        def loss_fn(model, outputs, labels):
            return self.criterion(outputs, labels)

        single_setup = (self.model, self.defs, self.criterion, self.optimizer, self.scheduler)
        for self.epoch in range(max_epoch):
            self._step(kettle, poison_delta, loss_fn, self.epoch, stats, *single_setup)
            if self.args.dryrun:
                break
        return stats

    def step(self, kettle, poison_delta, poison_targets, true_classes):
        """Step through a model epoch. Optionally: minimize target loss."""
        stats = defaultdict(list)

        def loss_fn(model, outputs, labels):
            normal_loss = self.criterion(outputs, labels)
            model.eval()
            if self.args.adversarial != 0:
                target_loss = 1 / self.defs.batch_size * self.criterion(model(poison_targets), true_classes)
            else:
                target_loss = 0
            model.train()
            return normal_loss + self.args.adversarial * target_loss

        single_setup = (self.model, self.criterion, self.optimizer, self.scheduler)
        self._step(kettle, poison_delta, loss_fn, self.epoch, stats, *single_setup)
        self.epoch += 1
        if self.epoch > self.defs.epochs:
            self.epoch = 0
            print('Model reset to epoch 0.')
            self.model, self.criterion, self.optimizer, self.scheduler = self._initialize_model()
            self.model.to(**self.setup)
            if torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(self.model)
        return stats

    """ Various Utilities."""

    def eval(self, dropout=False):
        """Switch everything into evaluation mode."""
        def apply_dropout(m):
            """https://discuss.pytorch.org/t/dropout-at-test-time-in-densenet/6738/6."""
            if type(m) == torch.nn.Dropout:
                m.train()
        self.model.eval()
        if dropout:
            self.model.apply(apply_dropout)

    def reset_learning_rate(self):
        """Reset scheduler object to initial state."""
        _, _, self.optimizer, self.scheduler = self._initialize_model()

    def gradient(self, images, labels, criterion=None):
        """Compute the gradient of criterion(model) w.r.t to given data."""
        if criterion is None:
            loss = self.criterion(self.model(images), labels)
        else:
            loss = criterion(self.model(images), labels)
        gradients = torch.autograd.grad(loss, self.model.parameters(), only_inputs=True)
        grad_norm = 0
        for grad in gradients:
            grad_norm += grad.detach().pow(2).sum()
        grad_norm = grad_norm.sqrt()
        return gradients, grad_norm
    
    def sharp_grad(self, criterion, images, labels, sigma):
        """compute gradient of sharpness on clean training, return a tuple, return -1*grad"""
        import copy
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        inputs, targets = images.to(device), labels.to(device)
        net_clone = copy.deepcopy(self.model)
        add_gaussian(net_clone, sigma)
        output_p = net_clone(inputs)
        loss_s = criterion(output_p, targets)
        loss_grad = torch.autograd.grad(loss_s, net_clone.parameters(), only_inputs=True)
        loss_grad_list = list(loss_grad)
        for _ in range(19):
            net_clone = copy.deepcopy(self.model)
            add_gaussian(net_clone, sigma)
            output_p = net_clone(inputs)
            loss_s = criterion(output_p, targets)
            grad = torch.autograd.grad(loss_s, net_clone.parameters(), only_inputs=True)
            loss_grad_list = list_plus(loss_grad_list, list(grad))
        loss_grad_list = list_multiply(loss_grad_list, 1/20)
        total_grad = tuple(loss_grad_list) # transform to tuple
        grad_norm = 0
        for grad in total_grad:
            grad_norm += grad.detach().pow(2).sum()
        grad_norm = grad_norm.sqrt()
        return total_grad, grad_norm
    
    def worst_sharp_grad(self, criterion, images, labels, sigma):
        """grad of worst sharpness"""
        import copy
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        inputs, targets = images.to(device), labels.to(device)
        output = self.model(inputs)
        loss = criterion(output, targets)
        grad_ = torch.autograd.grad(loss, self.model.parameters(), only_inputs=True)
        grad_n = 0
        for grad in grad_:
            grad_n += grad.detach().pow(2).sum()
        grad_n = grad_n.sqrt()
        
        scale = sigma / (grad_n + 1e-12)
        net_clone = copy.deepcopy(self.model)
        for name, p in net_clone.named_parameters():
            if p.grad is None: continue
            e_w = (1.0) * p.grad * scale.to(p)
            p.add_(e_w)
        output_p = net_clone(inputs)
        loss_s = criterion(output_p, targets)
        gradients = torch.autograd.grad(loss_s, net_clone.parameters(), only_inputs=True)
        grad_norm = 0
        for grad in gradients:
            grad_norm += grad.detach().pow(2).sum()
        grad_norm = grad_norm.sqrt()
        return gradients, grad_norm

    def compute(self, function, *args):
        r"""Compute function on the given optimization problem, defined by criterion \circ model.

        Function has arguments: model, criterion
        """
        return function(self.model, self.criterion, self.optimizer, *args)
