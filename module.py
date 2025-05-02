import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import pytorch_lightning as pl
import pytorch_lightning.utilities as U
from schedule import cosine_lr_lambda
from functools import partial
from printer import print0
from zerocot_algo import *
import os, sys, time
import json
import codecs
import warnings

# 定义 LightningModule
class HFLM(pl.LightningModule):
    def __init__(self, model, base_model, dataset, tokenizer, args):
        super().__init__()
        self.args = args
        self.dataset = dataset
        self.tokenizer = tokenizer

        def set_attr_dict(s, d):
            if not d: return
            d = json.loads(d)
            assert isinstance(d, dict)
            for k, v in d.items():
                if hasattr(s, k):
                    setattr(s, k, v)
                    print0(f"Model {s} attribute `{k}` set to `{v}`!")

        set_attr_dict(model, args.model_args)
        set_attr_dict(model.config, args.config_args)
        model.train()
        base_model.eval()
        model.gradient_checkpointing_enable()
        
        if args.compile_model:
            warnings.warn("Compiling is experimental")
            self.model = torch.compile(model)
        else:
            self.model = model
        if args.compile_base_model:
            warnings.warn("Compiling is experimental")
            self.base_model = torch.compile(base_model)
        else:
            self.base_model = base_model
        

        think_token = self.tokenizer.encode(codecs.decode(args.think_token, 'unicode_escape'))
        stop_token = self.tokenizer.encode(codecs.decode(args.stop_token, 'unicode_escape'))
        
        assert len(think_token) == 1
        assert len(stop_token) == 1
        self.think_token = think_token[0]
        self.stop_token = stop_token[0]
        print0('think token <think> =', think_token)
        print0('stop token </think> =', stop_token)

        self.loss_func = torch.nn.CrossEntropyLoss(reduction='none')
        self.reward_process_func = torch.nn.LeakyReLU(negative_slope=args.negative_reward_slope)
        self.reward_process_func.eval()

        # Initialize variables for throughput calculation and EMA smoothing
        self.ema_throughput = 0.0  # Exponential Moving Average of throughput
        self.last_time = time.time()  # Timestamp for measuring time per step

    def training_step(self, batch, batch_idx):
        # batch: (B, L+1)
        self.model.eval()
        with torch.no_grad():
            # 1. explore (rollout) sampling
            output_tokens, thinking_mask = zerocot_sampling(
                self.model,
                batch,
                think_token=self.think_token,
                stop_token=self.stop_token,
                length=self.args.ctx_len
            )
            # decoded_batch = self.tokenizer.batch_decode(batch)
            # for s in decoded_batch:
            #     print(s)
            #     print('\n' + '-'*80 + '\n')
            # decoded_output = self.tokenizer.batch_decode(output_tokens)
            # for s in decoded_output:
            #     print(s
            #           .replace(codecs.decode(self.args.think_token, 'unicode_escape'), '\n<t>')
            #           .replace(codecs.decode(self.args.stop_token, 'unicode_escape'), '</t>\n')
            #     )
            #     print('\n' + '-'*80 + '\n')
            
            # gather non-thinking:
            # 2. extract reading
            B, L = thinking_mask.shape
            self.log('think_proportion', thinking_mask.sum() / (B*L))
            self.log('avg_think_switch', (thinking_mask[:, 1:] > thinking_mask[:, :-1]).sum() / B)


            reading_mask = ~thinking_mask # (B, L) L=ctxlen+1
            reading_lengths = torch.sum(reading_mask, dim=1) # (B, )
            base_reading_part, base_mask = zerocot_truncate_left_align(batch, reading_lengths) # (B, L0+1)

            # 3. base forward
            base_lm_output = self.base_model(
                input_ids=base_reading_part[:, :-1], # (B, L0)
                attention_mask=base_mask[:, :-1], # (B, L0)
                past_key_values=None,
                use_cache=False,
                output_hidden_states=False,
            )
            base_lm_output_logits = base_lm_output['logits'].to(torch.float32)
            base_lm_labels = base_reading_part[:, 1:]
            B, T0 = base_lm_labels.shape
            BT0 = B * T0
            base_nll = self.loss_func(
                base_lm_output_logits.view(BT0, -1),
                base_lm_labels.flatten(),
            ).view(B, T0)
            # base_nll_mask = base_mask[:, 1:].contiguous() # 大概用不上了……
        
        self.model.train()
        # now requiring grad!
        # 4. model forward, compute nll
        actor_lm_output = self.model(
            input_ids=output_tokens[:, :-1],
            past_key_values=None,
            use_cache=False,
            output_hidden_states=False,
        ) # (B, L))
        actor_lm_output_logits = actor_lm_output['logits'].to(torch.float32)
        reading_mask_BT = reading_mask[:, 1:].contiguous()
        no_think_when_read_nll, token_given_no_think_nll, token_when_think_nll = zerocot_compute_nll(
            actor_lm_output_logits, output_tokens[:, 1:], reading_mask_BT, 
            think_token=self.think_token, stop_token=self.stop_token
        )
        with torch.no_grad():
            # 5. compute reward
            actor_sl_nll = token_given_no_think_nll.detach().clone()
            reward_BT = zerocot_reward_fill(
                base_nll_bt0=base_nll,
                actor_nll_bt=actor_sl_nll,
                actor_read_mask_bt=reading_mask_BT,
                thinking_default_reward=self.args.thinking_default_reward
            )
            # 5': compensation
            self.log('total_reward_avg', reward_BT.mean())
            read_reward_avg = ((reward_BT*reading_mask_BT).sum() / reading_mask_BT.sum())
            self.log('read_reward_avg', read_reward_avg)
            if hasattr(self, 'reward_ema'):
                self.reward_ema = (self.args.reward_ema_delay) * self.reward_ema + (
                                1-self.args.reward_ema_delay) * read_reward_avg
            else:
                self.reward_ema = read_reward_avg
            self.log('read_reward_ema', self.reward_ema)
            self.log('read_reward_adv', read_reward_avg - self.reward_ema)
            reward_BT -= self.args.reward_ema_compensate * self.reward_ema

            # 5'': fix: add some reward to switch
            reading_mask_BT_before = reading_mask[:, :-1].contiguous()
            switched_mode = (reading_mask_BT != reading_mask_BT_before)
            reward_BT -= self.args.switch_reward * switched_mode

            processed_reward = self.reward_process_func(reward_BT)
            self.log('processed_reward_avg', processed_reward.mean())
            self.log('processed_read_reward_avg', (processed_reward*reading_mask_BT).sum() / reading_mask_BT.sum())
            # 6. compute Q
            Q_values = zerocot_discounted_cumsum(processed_reward, gamma=self.args.rl_gamma)
            Q_values.requires_grad_(False)
            Q_values *= self.args.rl_coeff

        # 7. compute loss
        # SL part: token_given_no_think_nll weight 1
        # RL part: no_think_when_read_nll and token_when_think_nll weight Q
        loss_unreduced = torch.where(
            reading_mask_BT,
            token_given_no_think_nll + Q_values * no_think_when_read_nll,
            Q_values * token_when_think_nll,
        )
        loss = loss_unreduced.nanmean()
        self.log("loss", loss, prog_bar=True, logger=True)
        self.log("lr", self.lr_schedulers().get_lr()[0])
        return loss

    def on_train_epoch_start(self):
        self.dataset.global_rank = self.global_rank
        self.dataset.real_epoch = self.current_epoch
        self.dataset.world_size = self.trainer.world_size
    
    def on_before_optimizer_step(self, optimizer):
        grad_norms = U.grad_norm(self, norm_type=2.0)
        self.log_dict({k: grad_norms[k] for k in grad_norms if 'total' in k})

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Compute percentage of training completed
        percent_completed = self.global_step / self.args.total_steps
        self.log("percent", percent_completed)

        # Measure time per step
        current_time = time.time()
        time_this_step = current_time - self.last_time
        self.last_time = current_time 

        # Compute raw throughput (tokens per second)
        tokens_per_step = self.args.real_bsz * self.args.ctx_len
        raw_throughput = tokens_per_step / time_this_step

        self.ema_throughput = 0.9 * self.ema_throughput + 0.1 * raw_throughput
        self.log("throughput", self.ema_throughput, prog_bar=True, logger=True)

        if self.ema_throughput > 0 and self.trainer.global_step >= 40:
            remaining_steps = self.args.total_steps - self.global_step
            remaining_tokens = remaining_steps * tokens_per_step
            eta_seconds = remaining_tokens / self.ema_throughput
            eta_minutes = eta_seconds / 60  # Convert seconds to minutes
            self.log("eta", eta_minutes, prog_bar=True, logger=True)
        
        if self.global_step >= self.args.total_steps:
            self.model.save_pretrained(
                os.path.join(self.args.save_path, f"steps_{self.args.total_steps}_final")
            )
            sys.exit(0)

    def configure_optimizers(self):
        def is_matrix(shape):
            prod = 1
            for d in shape:
                if d >= 8:
                    prod *= d
            return prod >= 16384

        optim_groups = []
        # 这里需要对于1D和0D的单列出来
        if self.args.weight_decay > 0 and self.args.no_decay_1d:
            decay_set = []
            nodecay_set = []
            for n, p in self.model.named_parameters():
                print0(f"{n:60}", end='')
                add_decay = is_matrix(p.shape)
                print0(f"{int(add_decay)}   ", list(p.shape))
                if add_decay:
                    decay_set.append(p)
                else:
                    nodecay_set.append(p)
            optim_groups = [
                {"params": decay_set, "weight_decay": self.args.weight_decay},
                {"params": nodecay_set, "weight_decay": 0.0},
            ]

        # if 'deepspeed' in self.args.strategy:
        from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
        optimizer = (DeepSpeedCPUAdam if 'offload' in self.args.strategy else FusedAdam)(
            optim_groups if optim_groups else self.model.parameters(),
            lr=self.args.lr,
            betas=(self.args.beta1, self.args.beta2),
            weight_decay=self.args.weight_decay,
            eps=self.args.epsilon,
        )

        # 使用 LambdaLR 实现自定义调度器
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, 
            lr_lambda=partial(
                cosine_lr_lambda, 
                total_steps=self.args.total_steps, warmup_steps=self.args.warmup_steps,
                lr=self.args.lr, lr_end=self.args.lr_end,
            )
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # 按 step 更新学习率
                "frequency": 1       # 每个 step 都更新
            }
        }
    
    def train_dataloader(self):
        self.dataset.global_rank = self.global_rank
        self.dataset.real_epoch = self.current_epoch
        self.dataset.world_size = self.trainer.world_size
        return torch.utils.data.DataLoader(
            self.dataset,
            shuffle=False,
            pin_memory=('gpu' in self.args.accelerator.lower()),
            batch_size=self.args.bsz,
            num_workers=1,
            persistent_workers=False,
            drop_last=True
        )