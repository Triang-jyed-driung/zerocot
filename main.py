import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import no_init_weights
# from torch.utils.data import DataLoader
from dataset import MyDataset
from module import HFLM
# from callbacker import Callbacker
from pytorch_lightning.callbacks import ModelCheckpoint
import codecs
pl.seed_everything(0)
torch.set_float32_matmul_precision('high')
torch.set_printoptions(precision=2, sci_mode=False)
# 参数解析
parser = argparse.ArgumentParser()

# 模型相关参数
parser.add_argument("--model_name", type=str, required=True, help="Path to the Model")
parser.add_argument("--base_model_name", type=str, required=True, help="Path to the Base Model")
parser.add_argument("--model_args", type=str,
                    default="", help="Model args. e.g. {\"_attn_implementation\":flash_attention_2}")
parser.add_argument("--config_args", type=str,
                    default="", help="config args")
parser.add_argument("--compile_model", action="store_true", 
                    default=False, help="apply torch.compile to the model")
parser.add_argument("--compile_base_model", action="store_true", 
                    default=False, help="apply torch.compile to the base model")

# 数据集相关参数
parser.add_argument("--data_file", type=str, required=True, help="Path to (.bin, pretokenized) file")
parser.add_argument("--data_dtype", type=str, default="uint16", help="Data type of the dataset")
parser.add_argument("--ctx_len", type=int, default=2048, help="Context length")

# zerocot相关参数
parser.add_argument("--think_token", type=str, 
                    default="\\x17", help="The token for <think>, supports escape str")
parser.add_argument("--stop_token", type=str, 
                    default="\\x19", help="The token for </think>, supports escape str")
parser.add_argument("--rl_gamma", type=float, 
                    default=0.95, help="RL gamma, discount factor")
parser.add_argument("--rl_coeff", type=float, 
                    default=0.05, help="rl weight (1-gamma?)")
parser.add_argument("--negative_reward_slope", type=float, 
                     default=1.0, help="negative reward slope (<1 to encourage RL)")
parser.add_argument("--thinking_default_reward", type=float, 
                     default=0.0, help="reward for each thinking token")
parser.add_argument("--switch_reward", type=float, 
                     default=0.0, help="switching mode deserves how much reward")
parser.add_argument("--reward_ema_delay", type=float, 
                     default=0.9, help="reward ema delay (variance reduction somehow & avoids thinking degeneration)")
parser.add_argument("--reward_ema_compensate", type=float, 
                     default=1.0, help="reward ema compensation")
# thinking_default_reward
# parser.add_argument("--reward_clamp_up", type=float, 
#                     default=5.0, help="reward clamping upper bound")
# parser.add_argument("--reward_clamp_down", type=float, 
#                     default=-5.0, help="reward clamping lower bound")


# data sampling
parser.add_argument("--my_data_shift", type=int, default=1, help="Data shift")
parser.add_argument("--bsz", type=int, default=8, help="micro batch size")
parser.add_argument("--steps_per_epoch", type=int, default=11520, 
                    help="how many total steps for an epoch")

# 优化相关参数
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate at the start")
parser.add_argument("--lr_end", type=float, default=1e-5, help="Learning rate at the end")
parser.add_argument("--warmup_steps", type=int, default=0, help="Warmup steps")
parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1")
parser.add_argument("--beta2", type=float, default=0.99, help="AdamW beta2")
parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay")
parser.add_argument("--epsilon", type=float, default=2**(-32), help="AdamW epsilon")
parser.add_argument("--no_decay_1d", action="store_true", 
                    default=False, help="no decay on vectors and scalars")
parser.add_argument("--grad_clip_val", type=float, default=1.0, help="grad clip val")

# 训练相关参数
parser.add_argument("--precision", type=str, default="bf16", help="Precision for mixed precision training")
parser.add_argument("--accelerator", type=str, default="gpu", help="Accelerator type (gpu/cpu)")
parser.add_argument("--devices", type=int, default=1, help="Number of devices")
parser.add_argument("--strategy", type=str, default="deepspeed_stage_1", help="strategy (ddp, deepspeed)")

# WandB 相关参数
parser.add_argument("--wandb", type=str, default="MyTrainer", help="WandB project name")
parser.add_argument("--log_every", type=int, default=1, help="log every n steps")

# 保存相关参数
# parser.add_argument("--no_save", action="store_true", default=False, help="don't save")
parser.add_argument("--save_path", type=str, default="./out", help="Path to save pretrained")
parser.add_argument("--save_every", type=int, default=1, help="save every n epochs")
# parser.add_argument("--save_optim", action="store_true", 
#                     default=False, help="also save optimizers")

args = parser.parse_args()

os.makedirs(args.save_path, exist_ok=True)

# 加载数据集
# 假的epoch，方便管理
args.real_bsz = args.devices * args.bsz
args.samples_per_epoch = args.steps_per_epoch * args.real_bsz

dataset = MyDataset(args)

args.total_tokens = dataset.data_size
args.magic_prime = dataset.magic_prime
args.epoch_count = args.magic_prime // args.samples_per_epoch
args.total_steps = args.magic_prime // args.real_bsz

# 加载模型
model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)
model.train()
base_model = AutoModelForCausalLM.from_pretrained(args.base_model_name, trust_remote_code=True)
base_model.eval()
tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
# data_loader = DataLoader(dataset, shuffle=False, pin_memory=('gpu' in args.accelerator.lower()), 
#                          batch_size=args.bsz, num_workers=1, persistent_workers=False, drop_last=True)


# 初始化 LightningModule
lightning_module = HFLM(model, base_model, dataset, tokenizer, args)

# 初始化 WandB Logger
wandb_logger = WandbLogger(
    project=args.wandb,
)
# 初始化回调
# callbacker = Callbacker(args)

def find_latest_checkpoint(save_path):
    if not os.path.exists(save_path):
        return None
    checkpoint_files = [f for f in os.listdir(save_path) if f.endswith(".ckpt")]
    if not checkpoint_files:
        return None
    latest_checkpoint = max(
        [os.path.join(save_path, f) for f in checkpoint_files],
        key=os.path.getmtime
    )
    return latest_checkpoint

latest_checkpoint = find_latest_checkpoint(args.save_path)
checkpointer = ModelCheckpoint(
    dirpath=args.save_path,
    every_n_epochs=args.save_every,
    save_top_k=-1,
    save_weights_only=False,
)

# class CustomStrategyWrapper:
#     def __init__(self, args, strategy):
#         self.args = args
#         self._strategy = strategy

#     @property
#     def lightning_restore_optimizer(self) -> bool:
#         """Override to disable restoring optimizers and schedulers."""
#         return self.args.save_optim

#     def __getattr__(self, name):
#         """Delegate all other attributes/methods to the wrapped strategy."""
#         return getattr(self._strategy, name)


# 初始化 Trainer
sampler = {
    ('use_distributed_sampler' if pl.__version__[0] == '2' else 'replace_sampler_ddp'): False
}
trainer = pl.Trainer(
    accelerator=args.accelerator,
    devices=args.devices,
    precision=args.precision,
    logger=wandb_logger,
    log_every_n_steps=args.log_every,
    strategy=args.strategy,
    callbacks=[checkpointer],
    max_epochs=-1,
    deterministic=True,
    gradient_clip_val=args.grad_clip_val,
    **sampler,
)
# 开始训练

trainer.fit(lightning_module, ckpt_path=latest_checkpoint)
