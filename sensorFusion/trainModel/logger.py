from abc import ABCMeta, abstractclassmethod
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

class _Logger(metaclass=ABCMeta):
    @abstractclassmethod
    def __init__(self):
        pass

    @abstractclassmethod
    def log_init(self, trainer):
        pass

    @abstractclassmethod
    def log_step(self):
        pass

    @abstractclassmethod
    def step(self, trainer):
        pass

    @abstractclassmethod
    def log_epoch(self):
        pass

def _find_record_params(instance):
    try:
        float(instance)
        is_float = True
    except (ValueError, TypeError):
        is_float = False
    if not hasattr(instance, 'record_params'):
        if is_float:
            return {'': float(instance)}
        else:
            return {}
    else:
        all_params = {}
        if is_float:
            all_params[type(instance).__name__: float]
        for rp in instance.record_params:
            param_list = _find_record_params(getattr(instance, rp))
            for key in list(param_list.keys()):
                if key == '':
                    param_list[rp] = param_list.pop(key)
                else:
                    param_list[rp + '.' + key] = param_list.pop(key)
            all_params.update(param_list)
        return all_params

class TensorBoardLogger(_Logger):
    def __init__(self, tb_log_dir, comment='', log_interval=10, hparams={}):
        self.log_df = pd.DataFrame()
        self.log_interval = log_interval
        self.globa_step_counter = 0
        self.epoch_counter = 0
        self.tb_log_dir = tb_log_dir
        self.tb_writer = SummaryWriter(self.tb_log_dir, comment=comment)
        if hparams:
            self.tb_writer.add_hparams(hparams)
        # trainer as member or parameter?

    def log_init(self, trainer):
        self.steps_per_epoch = trainer.steps_per_epoch

    def step(self, trainer):
        record_params = _find_record_params(trainer)
        self.log_df = self.log_df.append(record_params, ignore_index=True)
        if (self.globa_step_counter + 1) % self.log_interval == 0:
            self.log_step()
        if (self.globa_step_counter + 1) % self.steps_per_epoch == 0:
            self.epoch_counter = trainer.epoch_num
            self.log_epoch()
        self.globa_step_counter += 1

    def log_step(self):
        step_mean = self.log_df[-self.log_interval:].mean()
        for k, v in step_mean.items():
            if v is not None:
                self.tb_writer.add_scalar('step/' + k, v, self.globa_step_counter)

    def log_epoch(self):
        epoch_mean = self.log_df[-self.steps_per_epoch:].mean()
        for k, v in epoch_mean.items():
            if v is not None:
                self.tb_writer.add_scalar('epoch/' + k, v, self.epoch_counter)