import torch
import torch.nn.functional as F
import torch.optim as optim

from torch import autograd
from torch.utils.data import DataLoader

from tqdm import tqdm


class ElasticWeightConsolidation:
    def __init__(self, model, lr=0.0001, weight=1000000):
        self.model = model.to('cuda')
        self.model.eval()

        self.weight = weight
        self.crit = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr)

        self.accuracies = {'task1': [], 'task2': []}

    @staticmethod
    def _target_offset(task):
        return (task - 1) * 10

    def _update_mean_params(self):
        for param_name, param in tqdm(self.model.named_parameters(), desc='Mean'):
            _buff_param_name = param_name.replace('.', '__')
            self.model.register_buffer(f'{_buff_param_name}_estimated_mean', param.data.clone())

    def _update_fisher_params(self, current_ds, batch_size, num_batch):
        dl = DataLoader(current_ds, batch_size, shuffle=True)
        log_liklihoods = []
        for i, (input, target) in tqdm(enumerate(dl), desc='Variance'):
            if i >= num_batch:
                break
            output = F.log_softmax(self.model(input.to('cuda')), dim=1)
            log_liklihoods.append(output[:, target.to('cuda')])
        log_likelihood = torch.cat(log_liklihoods).mean()
        grad_log_likelihood = autograd.grad(log_likelihood, self.model.parameters())
        _buff_param_names = [param[0].replace('.', '__') for param in self.model.named_parameters()]
        for _buff_param_name, param in zip(_buff_param_names, grad_log_likelihood):
            self.model.register_buffer(f'{_buff_param_name}_estimated_fisher', param.data.clone() ** 2)

    def register_ewc_params(self, dataset, batch_size, num_batches):
        self._update_mean_params()
        self._update_fisher_params(dataset, batch_size, num_batches)

    def _compute_consolidation_loss(self, weight):
        try:
            losses = []
            for param_name, param in self.model.named_parameters():
                _buff_param_name = param_name.replace('.', '__')
                estimated_mean = getattr(self.model, f'{_buff_param_name}_estimated_mean')
                estimated_fisher = getattr(self.model, f'{_buff_param_name}_estimated_fisher')

                if False and param_name.startswith('lin3'):
                    # Do not apply weight consolidation to weights related only to new task
                    task2_offset = self._target_offset(task=2)
                    mask = torch.tensor([1] * task2_offset + [0] * task2_offset, dtype=torch.float32, device='cuda')
                    if 'bias' in param_name:
                        estimated_fisher *= mask
                    else:
                        estimated_fisher *= mask[:, None]

                losses.append((estimated_fisher * (param - estimated_mean) ** 2).sum())
            return (weight / 2) * sum(losses)
        except AttributeError:
            return 0

    def _forward_backward_update(self, input, target):
        self.model.train()
        output = self.model(input.to('cuda'))
        self.optimizer.zero_grad()
        loss = self._compute_consolidation_loss(self.weight) + self.crit(output, target.to('cuda'))
        loss.backward()
        self.optimizer.step()
        self.model.eval()

    def accuracy(self, dataloader, task):
        acc = 0
        target_offset = self._target_offset(task)
        for input, target in dataloader:
            o = self.model(input.to('cuda')).cpu()
            acc += (o.argmax(dim=1).long() == target + target_offset).float().mean()
        return float(acc / len(dataloader))

    def train(self, train_dataloader, test_dataloaders, epochs, task):
        for i in (1, 2):
            if len(self.accuracies[f'task{i}']) == 0:
                self.accuracies[f'task{i}'].append(self.accuracy(test_dataloaders[i - 1], task=i))

        target_offset = self._target_offset(task)
        for epoch in range(epochs):
            for input, target in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}'):
                self._forward_backward_update(input, target + target_offset)

            for i in (1, 2):
                self.accuracies[f'task{i}'].append(self.accuracy(test_dataloaders[i - 1], task=i))
