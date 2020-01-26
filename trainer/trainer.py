import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from base import BaseTrainer
import dgl

class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, pre_metric, optimizer, config,
                 data_loader, valid_data_loader=None, lr_scheduler=None):
        super(Trainer, self).__init__(model, loss, metrics, optimizer, config)
        self.config = config
        self.readout_method = model.readout_method
        self.is_infonce_training = config['loss'].startswith("info_nce")
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_mode = self.config['lr_scheduler']['args']['mode']  # "min" or "max"
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.pre_metric = pre_metric
        self.writer.add_text('Text', 'Model Architecture: {}'.format(self.config['arch']), 0)
        self.writer.add_text('Text', 'Training Data Loader: {}'.format(self.config['train_data_loader']), 0)
        self.writer.add_text('Text', 'Loss Function: {}'.format(self.config['loss']), 0)
        self.writer.add_text('Text', 'Optimizer: {}'.format(self.config['optimizer']), 0)

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        all_ranks = self.pre_metric(output, target)
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(all_ranks)
            self.writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch_idx, batch_example in enumerate(self.data_loader):
            bg = batch_example[0]
            nf = batch_example[1].to(self.device)
            label = batch_example[2].to(self.device)
            h = bg.ndata.pop('x').to(self.device)
            
            self.optimizer.zero_grad()
            prediction = self.model(bg, h, nf)
            if self.is_infonce_training:
                n_batches = label.sum().detach()
                prediction = prediction.reshape(n_batches, -1)
                target = torch.zeros(n_batches, dtype=torch.long).to(self.device)
                loss = self.loss(prediction, target)
            else:
                loss = self.loss(prediction, label)

            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
            self.writer.add_scalar('loss', loss.item())
            total_loss += loss.item()

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch,
                    batch_idx * self.data_loader.batch_size,
                    self.data_loader.n_samples,
                    100.0 * batch_idx / len(self.data_loader),
                    loss.item()))

        log = {
            'loss': total_loss / len(self.data_loader),
        }

        ## Validation stage
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                print(f"log['val_metrics]: {log['val_metrics']}")
                if self.lr_scheduler_mode == "min":
                    self.lr_scheduler.step(log['val_metrics'][0])  # TODO: for MAG-CS, early stop on MR
                else:
                    self.lr_scheduler.step(log['val_metrics'][2])  # TODO: for SemEval, early stop on Hit@1
            else:
                self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for batch_idx, batch_example in enumerate(self.valid_data_loader):
                bg = batch_example[0]
                qf = batch_example[1].to(self.device)
                label = batch_example[2].to(self.device)
                h = bg.ndata.pop('x').to(self.device)
                prediction = self.model(bg, h, qf)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                total_val_metrics += self._eval_metrics(prediction, label)

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        return {
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }
