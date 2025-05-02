import pytorch_lightning as pl

# 定义一个回调类，用于在每个 epoch 开始时更新数据集属性
class Callbacker(pl.Callback):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def on_train_epoch_start(self, trainer, pl_module):
        # 获取当前 epoch 和分布式训练相关的信息
        if pl.__version__[0] == '2':
            d = trainer.train_dataloader.dataset
        else:
            d = trainer.train_dataloader.dataset.datasets
        d.global_rank = trainer.global_rank
        d.real_epoch = trainer.current_epoch
        d.world_size = trainer.world_size

    # def on_load_checkpoint(self, trainer, pl_module, checkpoint):
    #     checkpoint["optimizer_states"] = []

    # def on_save_checkpoint(self, trainer, pl_module, checkpoint):
    #     if 'optimizer_states' in checkpoint and not self.args.save_optim:
    #         del checkpoint['optimizer_states']