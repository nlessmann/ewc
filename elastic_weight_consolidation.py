import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
from torch.utils.data import DataLoader


class ElasticWeightConsolidation:

    def __init__(self, model, crit, lr=0.001, weight=1000000):
        self.model = model.to('cuda')
        self.model.train(False)

        self.weight = weight
        self.crit = crit
        self.optimizer = optim.SGD(self.model.parameters(), lr)

    def _update_mean_params(self):
        for param_name, param in self.model.named_parameters():
            _buff_param_name = param_name.replace('.', '__')
            self.model.register_buffer(_buff_param_name+'_estimated_mean', param.data.clone())

    def _update_fisher_params(self, current_ds, batch_size, num_batch):
        dl = DataLoader(current_ds, batch_size, shuffle=True)
        log_liklihoods = []
        for i, (input, target) in enumerate(dl):
            if i >= num_batch:
                break
            output = F.log_softmax(self.model(input.to('cuda')), dim=1)
            log_liklihoods.append(output[:, target.to('cuda')])
        log_likelihood = torch.cat(log_liklihoods).mean()
        grad_log_likelihood = autograd.grad(log_likelihood, self.model.parameters())
        _buff_param_names = [param[0].replace('.', '__') for param in self.model.named_parameters()]
        for _buff_param_name, param in zip(_buff_param_names, grad_log_likelihood):
            self.model.register_buffer(_buff_param_name+'_estimated_fisher', param.data.clone() ** 2)

    def register_ewc_params(self, dataset, batch_size, num_batches):
        self._update_fisher_params(dataset, batch_size, num_batches)
        self._update_mean_params()

    def _compute_consolidation_loss(self, weight):
        try:
            losses = []
            for param_name, param in self.model.named_parameters():
                _buff_param_name = param_name.replace('.', '__')
                estimated_mean = getattr(self.model, '{}_estimated_mean'.format(_buff_param_name))
                estimated_fisher = getattr(self.model, '{}_estimated_fisher'.format(_buff_param_name))

                if False and param_name.startswith('lin3'):
                    mask = torch.tensor([1] * 10 + [0] * 10, dtype=torch.float32, device='cuda')
                    if 'bias' in param_name:
                        estimated_fisher *= mask
                    else:
                        estimated_fisher *= mask[:, None]

                losses.append((estimated_fisher * (param - estimated_mean) ** 2).sum())

            return (weight / 2) * sum(losses)
        except AttributeError:
            return 0

    def forward_backward_update(self, input, target):
        self.model.train(True)
        output = self.model(input.to('cuda'))
        self.optimizer.zero_grad()
        loss = self._compute_consolidation_loss(self.weight) + self.crit(output, target.to('cuda'))
        loss.backward()
        self.optimizer.step()
        self.model.train(False)

    def save(self, filename):
        torch.save(self.model, filename)

    def load(self, filename):
        self.model = torch.load(filename)
