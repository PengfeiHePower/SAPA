"""Single model default victim class."""

from numpy.lib.type_check import imag
import torch
import numpy as np
import warnings
from math import ceil

from collections import defaultdict
import copy

from .models import get_model
from .training import get_optimizers
from ..hyperparameters import training_strategy
from ..utils import set_random_seed
from ..consts import BENCHMARK
torch.backends.cudnn.benchmark = BENCHMARK

from .victim_base import _VictimBase

def add_gaussian(model, sigma=0.05):
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

    def initialize(self, pretrain=False, seed=None):
        if self.args.modelkey is None:
            if seed is None:
                self.model_init_seed = np.random.randint(0, 2**32 - 1)
            else:
                self.model_init_seed = seed
        else:
            self.model_init_seed = self.args.modelkey
        set_random_seed(self.model_init_seed)
        self.model, self.defs, self.optimizer, self.scheduler = self._initialize_model(self.args.net[0], pretrain=pretrain)

        self.model.to(**self.setup)
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
            self.model.frozen = self.model.module.frozen
        print(f'{self.args.net[0]} model initialized with random key {self.model_init_seed}.')
        print(repr(self.defs))

    def reinitialize_last_layer(self, reduce_lr_factor=1.0, seed=None, keep_last_layer=False):
        if not keep_last_layer:
            if self.args.modelkey is None:
                if seed is None:
                    self.model_init_seed = np.random.randint(0, 2**32 - 1)
                else:
                    self.model_init_seed = seed
            else:
                self.model_init_seed = self.args.modelkey
            set_random_seed(self.model_init_seed)

            # We construct a full replacement model, so that the seed matches up with the initial seed,
            # even if all of the model except for the last layer will be immediately discarded.
            replacement_model = get_model(self.args.net[0], self.args.dataset, pretrained=self.args.pretrained_model)

            # Rebuild model with new last layer
            frozen = self.model.frozen
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1], torch.nn.Flatten(), list(replacement_model.children())[-1])
            self.model.frozen = frozen
            self.model.to(**self.setup)
            if torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(self.model)
                self.model.frozen = self.model.module.frozen

        # Define training routine
        # Reinitialize optimizers here
        self.defs = training_strategy(self.args.net[0], self.args)
        self.defs.lr *= reduce_lr_factor
        self.optimizer, self.scheduler = get_optimizers(self.model, self.args, self.defs)
        print(f'{self.args.net[0]} last layer re-initialized with random key {self.model_init_seed}.')
        print(repr(self.defs))

    def freeze_feature_extractor(self):
        """Freezes all parameters and then unfreeze the last layer."""
        self.model.frozen = True
        for param in self.model.parameters():
            param.requires_grad = False

        for param in list(self.model.children())[-1].parameters():
            param.requires_grad = True

    def save_feature_representation(self):
        self.clean_model = copy.deepcopy(self.model)

    def load_feature_representation(self):
        self.model = copy.deepcopy(self.clean_model)


    """ METHODS FOR (CLEAN) TRAINING AND TESTING OF BREWED POISONS"""

    def _iterate(self, kettle, poison_delta, max_epoch=None, pretraining_phase=False):
        """Validate a given poison by training the model and checking source accuracy."""
        stats = defaultdict(list)

        if max_epoch is None:
            max_epoch = self.defs.epochs

        single_setup = (self.model, self.defs, self.optimizer, self.scheduler)
        for self.epoch in range(max_epoch):
            self._step(kettle, poison_delta, self.epoch, stats, *single_setup, pretraining_phase)
            if self.args.dryrun:
                break
        return stats

    def step(self, kettle, poison_delta, poison_sources, true_classes):
        """Step through a model epoch. Optionally: minimize source loss."""
        stats = defaultdict(list)


        single_setup = (self.model, self.defs, self.optimizer, self.scheduler)
        self._step(kettle, poison_delta, self.epoch, stats, *single_setup)
        self.epoch += 1
        if self.epoch > self.defs.epochs:
            self.epoch = 0
            print('Model reset to epoch 0.')
            self.model, self.defs, self.optimizer, self.scheduler = self._initialize_model(self.args.net[0])
            self.model.to(**self.setup)
            if torch.cuda.device_count() > 1 and 'meta' not in self.defs.novel_defense['type']:
                self.model = torch.nn.DataParallel(self.model)
                self.model.frozen = self.model.module.frozen
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
        _, _, self.optimizer, self.scheduler = self._initialize_model(self.args.net[0])

    def gradient(self, images, labels, criterion=None, selection=None):
        """Compute the gradient of criterion(model) w.r.t to given data."""

        if criterion is None:
                criterion = self.loss_fn
        differentiable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        if selection == 'max_gradient':
            grad_norms = []
            for image, label in zip(images, labels):
                loss = criterion(self.model(image.unsqueeze(0)), label.unsqueeze(0))
                gradients = torch.autograd.grad(loss, differentiable_params, only_inputs=True)
                grad_norm = 0
                for grad in gradients:
                    grad_norm += grad.detach().pow(2).sum()
                grad_norms.append(grad_norm.sqrt())
            
            indices = [i[0] for i in sorted(enumerate(grad_norms), key=lambda x:x[1])][-self.args.num_source_selection:]
            images = images[indices]
            labels = labels[indices]
            print('{} sources with maximum gradients selected'.format(self.args.num_source_selection))


        # Using batch processing for gradients
        if not self.args.source_gradient_batch==None:
            batch_size = self.args.source_gradient_batch
            if images.shape[0] < batch_size:
                batch_size = images.shape[0]
            else:
                if images.shape[0] % batch_size != 0:
                    batch_size = images.shape[0] // ceil(images.shape[0] / batch_size)
                    warnings.warn(f'Batch size changed to {batch_size} to fit source train size')
            gradients = None
            for i in range(images.shape[0]//batch_size):
                loss = batch_size * criterion(self.model(images[i*batch_size:(i+1)*batch_size]), labels[i*batch_size:(i+1)*batch_size])
                if i == 0:
                    gradients = torch.autograd.grad(loss, differentiable_params, only_inputs=True)
                else:
                    gradients = tuple(map(lambda i, j: i + j, gradients, torch.autograd.grad(loss, differentiable_params, only_inputs=True)))

            gradients = tuple(map(lambda i: i / images.shape[0], gradients))
        else:
            loss = criterion(self.model(images), labels)
            gradients = torch.autograd.grad(loss, differentiable_params, only_inputs=True)

        grad_norm = 0
        for grad in gradients:
            grad_norm += grad.detach().pow(2).sum()
        grad_norm = grad_norm.sqrt()
    
        return gradients, grad_norm
    
    def worst_sharp_grad(self, images, labels, criterion=None, selection=None, sigma=0.05):
        import copy
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if criterion is None:
                criterion = self.loss_fn
        differentiable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        if selection == 'max_gradient':
            grad_norms = []
            for image, label in zip(images, labels):
                loss = criterion(self.model(image.unsqueeze(0)), label.unsqueeze(0))
                gradients = torch.autograd.grad(loss, differentiable_params, only_inputs=True)
                grad_norm = 0
                for grad in gradients:
                    grad_norm += grad.detach().pow(2).sum()
                grad_norms.append(grad_norm.sqrt())
            
            indices = [i[0] for i in sorted(enumerate(grad_norms), key=lambda x:x[1])][-self.args.num_source_selection:]
            images = images[indices]
            labels = labels[indices]
            print('{} sources with maximum gradients selected'.format(self.args.num_source_selection))
        
        if not self.args.source_gradient_batch==None:
            batch_size = self.args.source_gradient_batch
            if images.shape[0] < batch_size:
                batch_size = images.shape[0]
            else:
                if images.shape[0] % batch_size != 0:
                    batch_size = images.shape[0] // ceil(images.shape[0] / batch_size)
                    warnings.warn(f'Batch size changed to {batch_size} to fit source train size')
            grad_ = None
            for i in range(images.shape[0]//batch_size):
                loss = batch_size * criterion(self.model(images[i*batch_size:(i+1)*batch_size]), labels[i*batch_size:(i+1)*batch_size])
                if i == 0:
                    grad_ = torch.autograd.grad(loss, differentiable_params, only_inputs=True)
                else:
                    grad_ = tuple(map(lambda i, j: i + j, grad_, torch.autograd.grad(loss, differentiable_params, only_inputs=True)))

            grad_ = tuple(map(lambda i: i / images.shape[0], grad_))
        else:
            loss = criterion(self.model(images), labels)
            grad_ = torch.autograd.grad(loss, differentiable_params, only_inputs=True)

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
        
        if not self.args.source_gradient_batch==None:
            batch_size = self.args.source_gradient_batch
            if images.shape[0] < batch_size:
                batch_size = images.shape[0]
            else:
                if images.shape[0] % batch_size != 0:
                    batch_size = images.shape[0] // ceil(images.shape[0] / batch_size)
                    warnings.warn(f'Batch size changed to {batch_size} to fit source train size')
            gradients = None
            for i in range(images.shape[0]//batch_size):
                loss = batch_size * criterion(net_clone(images[i*batch_size:(i+1)*batch_size]), labels[i*batch_size:(i+1)*batch_size])
                if i == 0:
                    gradients = torch.autograd.grad(loss, net_clone.parameters(), only_inputs=True)
                else:
                    gradients = tuple(map(lambda i, j: i + j, gradients, torch.autograd.grad(loss, net_clone.parameters(), only_inputs=True)))

            gradients = tuple(map(lambda i: i / images.shape[0], gradients))
        else:
            loss = criterion(net_clone(images), labels)
            gradients = torch.autograd.grad(loss, net_clone.parameters(), only_inputs=True)
        
        grad_norm = 0
        for grad in gradients:
            grad_norm += grad.detach().pow(2).sum()
        grad_norm = grad_norm.sqrt()
        
        return gradients, grad_norm
    
    def sharp_grad(self, images, labels, criterion=None, selection=None, sigma=0.05):
        import copy
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if criterion is None:
                criterion = self.loss_fn
        differentiable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        if selection == 'max_gradient':
            grad_norms = []
            for image, label in zip(images, labels):
                loss = criterion(self.model(image.unsqueeze(0)), label.unsqueeze(0))
                gradients = torch.autograd.grad(loss, differentiable_params, only_inputs=True)
                grad_norm = 0
                for grad in gradients:
                    grad_norm += grad.detach().pow(2).sum()
                grad_norms.append(grad_norm.sqrt())
            
            indices = [i[0] for i in sorted(enumerate(grad_norms), key=lambda x:x[1])][-self.args.num_source_selection:]
            images = images[indices]
            labels = labels[indices]
            print('{} sources with maximum gradients selected'.format(self.args.num_source_selection))
        
        # Using batch processing for gradients
        if not self.args.source_gradient_batch==None:
            batch_size = self.args.source_gradient_batch
            if images.shape[0] < batch_size:
                batch_size = images.shape[0]
            else:
                if images.shape[0] % batch_size != 0:
                    batch_size = images.shape[0] // ceil(images.shape[0] / batch_size)
                    warnings.warn(f'Batch size changed to {batch_size} to fit source train size')
            gradients = None
            for i in range(images.shape[0]//batch_size):
                if i == 0:
                    net_clone = copy.deepcopy(self.model)
                    add_gaussian(net_clone, sigma)
                    output_p = net_clone(images[i*batch_size:(i+1)*batch_size])
                    loss_s = batch_size * criterion(output_p, labels[i*batch_size:(i+1)*batch_size])
                    gradients = torch.autograd.grad(loss_s, net_clone.parameters(), only_inputs=True)
                    for _ in range(19):
                        net_clone = copy.deepcopy(self.model)
                        add_gaussian(net_clone, sigma)
                        output_p = net_clone(images[i*batch_size:(i+1)*batch_size])
                        loss_s = batch_size * criterion(output_p, labels[i*batch_size:(i+1)*batch_size])
                        gradients = tuple(map(lambda i, j: i + j, gradients, torch.autograd.grad(loss_s, net_clone.parameters(), only_inputs=True)))
                    gradients = tuple(map(lambda i: i / 20, gradients))
                else:
                    net_clone = copy.deepcopy(self.model)
                    add_gaussian(net_clone, sigma)
                    output_p = net_clone(images[i*batch_size:(i+1)*batch_size])
                    loss_s = batch_size * criterion(output_p, labels[i*batch_size:(i+1)*batch_size])
                    gradients_m = torch.autograd.grad(loss_s, net_clone.parameters(), only_inputs=True)
                    for _ in range(19):
                        net_clone = copy.deepcopy(self.model)
                        add_gaussian(net_clone, sigma)
                        output_p = net_clone(images[i*batch_size:(i+1)*batch_size])
                        loss_s = batch_size * criterion(output_p, labels[i*batch_size:(i+1)*batch_size])
                        gradients_m = tuple(map(lambda i, j: i + j, gradients_m, torch.autograd.grad(loss_s, net_clone.parameters(), only_inputs=True)))
                    gradients_m = tuple(map(lambda i: i / 20, gradients_m))
                    gradients = tuple(map(lambda i, j: i + j, gradients, gradients_m))
            gradients = tuple(map(lambda i: i / images.shape[0], gradients))
        else:
            net_clone = copy.deepcopy(self.model)
            add_gaussian(net_clone, sigma)
            output_p = net_clone(images)
            loss_s = criterion(output_p, labels)
            gradients = torch.autograd.grad(loss_s, net_clone.parameters(), only_inputs=True)
            for _ in range(19):
                net_clone = copy.deepcopy(self.model)
                add_gaussian(net_clone, sigma)
                output_p = net_clone(images)
                loss_s = batch_size * criterion(output_p, labels)
                gradients = tuple(map(lambda i, j: i + j, gradients, torch.autograd.grad(loss_s, net_clone.parameters(), only_inputs=True)))
            gradients = tuple(map(lambda i: i / 20, gradients))

        grad_norm = 0
        for grad in gradients:
            grad_norm += grad.detach().pow(2).sum()
        grad_norm = grad_norm.sqrt()
        
        return gradients, grad_norm

    def compute(self, function, *args):
        r"""Compute function on the given optimization problem, defined by criterion \circ model.

        Function has arguments: model, criterion
        """
        return function(self.model, self.optimizer, *args)
