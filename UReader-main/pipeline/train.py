import argparse
import json
import os
from functools import partial

import torch
import torch.distributed as dist

from sconf import Config
from icecream import ic
from peft import LoraConfig, get_peft_model
from transformers.training_args import TrainingArguments

from mplug_owl import MplugOwlForConditionalGeneration, MplugOwlTokenizer
from pipeline.data_utils import train_valid_test_datasets_provider
from pipeline.trainer import CustomTrainer
from pipeline.utils import add_config_args, set_args

parser = argparse.ArgumentParser()
# Model
parser.add_argument('--pretrained-ckpt', type=str, default='MAGAer13/mplug-owl-llama-7b-pt',
                    help='Path to the pretrained checkpoint.')
parser.add_argument('--inference_mode', type=bool, default=False,
                    help='The inference mode.')
parser.add_argument('--seq-length', type=int, default=1024,
                    help='Maximum sequence length to process.')
parser.add_argument('--freeze-v2t', action='store_true', help='Freeze abstractor')
parser.add_argument('--language-training-method', type=str, default='lora', help='lora/training/none.')
parser.add_argument('--lora-r', type=int, default=8,
                    help='LoRA rank.')
parser.add_argument('--lora-alpha', type=int, default=32,
                    help='The scaling factor for LoRA.')
parser.add_argument('--lora-dropout', type=float, default=0.05,
                    help='Dropout for LoRA.')
parser.add_argument('--lora-target-modules', type=str, default='q_proj,v_proj',
                    help='Comma-separated target modules for language LoRA.')

# Pluggable LoRA switches
parser.add_argument('--lora-on-language', action='store_true',
                    help='Enable LoRA on language_model.')
parser.add_argument('--lora-on-abstractor', action='store_true',
                    help='Enable LoRA on abstractor (visual abstractor).')
parser.add_argument('--lora-on-vision', action='store_true',
                    help='Enable LoRA on vision_model.')
parser.add_argument('--abstractor-lora-target-modules', type=str, default='query,key,value,dense',
                    help='Comma-separated target modules for abstractor LoRA.')
parser.add_argument('--vision-lora-target-modules', type=str, default='query_key_value,dense,fc1,fc2',
                    help='Comma-separated target modules for vision LoRA.')
parser.add_argument('--save-lora-adapters-only', action='store_true',
                    help='Save LoRA adapters only if LoRA is enabled.')

parser.add_argument('--bf16', action='store_true', default=False,
                    help='Run model in bfloat16 mode.')

# Data
parser.add_argument('--mm-config', type=str, default='configs/sft/release.yaml', help='Multimodal Config.')
parser.add_argument('--image-root', type=str, default='ureader_images', help='Image folder.')
parser.add_argument('--num-workers', type=int, default=8,
                    help="Dataloader number of workers.")

# Training HyperParameters
parser.add_argument('--train-epochs', type=int, default=3,
                    help='Total number of epochs to train over all training runs.')
parser.add_argument('--micro-batch-size', type=int, default=None,
                    help='Batch size per model instance (local batch size).')
parser.add_argument('--global-batch-size', type=int, default=256,
                    help='Global batch size.')
parser.add_argument('--lr', type=float, default=None,
                    help='Initial learning rate.')
parser.add_argument('--min-lr', type=float, default=1e-6,
                    help='Minimum learning rate.')
parser.add_argument('--weight-decay', type=float, default=0.01,
                    help='Weight decay coefficient.')
parser.add_argument('--gradient-accumulation-steps', type=int, default=8,
                    help='Gradient accumulation steps.')
parser.add_argument('--clip-grad', type=float, default=1.0,
                    help='Gradient clipping based on global L2 norm.')
parser.add_argument('--adam-beta1', type=float, default=0.9)
parser.add_argument('--adam-beta2', type=float, default=0.999)
parser.add_argument('--adam-eps', type=float, default=1e-08)
parser.add_argument('--num-warmup-steps', type=int, default=50,
                    help='The number of warmup steps.')

# Evaluation & Save
parser.add_argument('--save-path', type=str, default=None,
                    help='Output directory to save checkpoints to.')
parser.add_argument('--save-interval', type=int, default=None,
                    help='Number of iterations between checkpoint saves.')
parser.add_argument('--eval-iters', type=int, default=100,
                    help='Number of iterations to run for evaluation.')

# Other
parser.add_argument('--gradient-checkpointing', action='store_true',
                    help='The gradient checkpointing.')
parser.add_argument('--logging-nan-inf-filter', action='store_true',
                    help='The logging nan inf filter.')
parser.add_argument('--ddp-find-unused-parameters', action='store_true',
                    help='unused parameters finding.')
parser.add_argument('--do-train', action='store_true', default=True,
                    help='Whether to do training.')
parser.add_argument('--local_rank', type=int, default=-1,
                    help='Local rank')

parser.add_argument('--tensorboard-dir', type=str)
parser.add_argument('--deepspeed', type=str, default=None)


def get_accumulation_step(args):
    global_batch_size = args.global_batch_size
    batch_size = args.micro_batch_size
    gpu_nums = dist.get_world_size()

    accumulation_step = max(1, int(round(global_batch_size / (batch_size * gpu_nums))))
    if accumulation_step * (batch_size * gpu_nums) != global_batch_size:
        import warnings
        warnings.warn(
            f"The actual global_batch_size is {accumulation_step * (batch_size * gpu_nums)} instead {global_batch_size}\n"
        )
    return accumulation_step


def parse_target_modules(spec: str):
    return [m.strip() for m in spec.split(',') if m.strip()]


def attach_lora(module, target_modules, args):
    peft_config = LoraConfig(
        target_modules=target_modules,
        inference_mode=args.inference_mode,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    return get_peft_model(module, peft_config)


def apply_pluggable_lora(model, args):
    """Apply LoRA adapters to selected modules in a pluggable manner."""
    enabled = {
        'language': args.lora_on_language,
        'abstractor': args.lora_on_abstractor,
        'vision': args.lora_on_vision,
    }

    # backward compatibility: old arg language-training-method=lora
    if args.language_training_method == 'lora' and not any(enabled.values()):
        enabled['language'] = True

    if enabled['language']:
        model.language_model = attach_lora(
            model.language_model,
            parse_target_modules(args.lora_target_modules),
            args,
        )
        model.language_model.print_trainable_parameters()

    if enabled['abstractor']:
        model.abstractor = attach_lora(
            model.abstractor,
            parse_target_modules(args.abstractor_lora_target_modules),
            args,
        )
        model.abstractor.print_trainable_parameters()

    if enabled['vision']:
        model.vision_model = attach_lora(
            model.vision_model,
            parse_target_modules(args.vision_lora_target_modules),
            args,
        )
        model.vision_model.print_trainable_parameters()

    return enabled


def save_lora_metadata(save_path, enabled, args):
    if not save_path:
        return
    os.makedirs(save_path, exist_ok=True)
    meta = {
        'pretrained_ckpt': args.pretrained_ckpt,
        'lora_enabled': enabled,
        'lora_r': args.lora_r,
        'lora_alpha': args.lora_alpha,
        'lora_dropout': args.lora_dropout,
        'lora_target_modules': parse_target_modules(args.lora_target_modules),
        'abstractor_lora_target_modules': parse_target_modules(args.abstractor_lora_target_modules),
        'vision_lora_target_modules': parse_target_modules(args.vision_lora_target_modules),
    }
    with open(os.path.join(save_path, 'lora_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main():
    args, left_argv = parser.parse_known_args()
    torch.distributed.init_process_group(backend="nccl")
    ic(left_argv)
    config = Config(args.mm_config)
    add_config_args(config, args)
    if args.global_batch_size is not None:
        args.gradient_accumulation_steps = get_accumulation_step(args)
    ic(args.gradient_accumulation_steps)
    set_args(args)

    model = MplugOwlForConditionalGeneration.from_pretrained(
        args.pretrained_ckpt,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float,
    )
    tokenizer = MplugOwlTokenizer.from_pretrained(args.pretrained_ckpt)

    # default freeze strategy
    for name, param in model.named_parameters():
        if 'vision_model' in name:
            param.requires_grad = False
        elif 'language_model' in name:
            param.requires_grad = False
        else:
            if args.freeze_v2t and ('query_tokens' in name or 'abstractor' in name):
                param.requires_grad = False
                continue
            param.requires_grad = True

    if args.language_training_method == 'training':
        for name, param in model.named_parameters():
            if 'language_model' in name:
                param.requires_grad = True

    enabled_lora = {'language': False, 'abstractor': False, 'vision': False}
    if args.language_training_method == 'lora' or args.lora_on_language or args.lora_on_abstractor or args.lora_on_vision:
        enabled_lora = apply_pluggable_lora(model, args)

    if args.gradient_checkpointing:
        model.vision_model.apply(partial(model.vision_model._set_gradient_checkpointing, value=True))
        ic(model.vision_model.encoder.gradient_checkpointing)

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.language_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        model.language_model.apply(partial(model.language_model._set_gradient_checkpointing, value=True))

    model.train()

    train_data, valid_data = train_valid_test_datasets_provider(
        config.data_files, config=config,
        tokenizer=tokenizer, seq_length=args.seq_length,
        image_root=args.image_root,
    )
    trainer = CustomTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=TrainingArguments(
            learning_rate=args.lr,
            warmup_steps=args.num_warmup_steps,
            do_train=args.do_train,
            num_train_epochs=args.train_epochs,
            output_dir=args.save_path,
            logging_dir=args.tensorboard_dir,
            save_strategy='steps',
            save_steps=args.save_interval,
            evaluation_strategy='steps',
            eval_steps=args.eval_iters,
            per_device_train_batch_size=args.micro_batch_size,
            max_grad_norm=args.clip_grad,
            weight_decay=args.weight_decay,
            bf16=args.bf16,
            fp16=not args.bf16,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_checkpointing=args.gradient_checkpointing,
            logging_steps=args.eval_iters // 4,
            logging_nan_inf_filter=args.logging_nan_inf_filter,
            ddp_find_unused_parameters=args.ddp_find_unused_parameters,
            deepspeed=args.deepspeed,
            dataloader_num_workers=args.num_workers,
        ),
    )

    trainer.train()

    if args.save_lora_adapters_only and any(enabled_lora.values()):
        os.makedirs(args.save_path, exist_ok=True)
        if enabled_lora['language']:
            model.language_model.save_pretrained(os.path.join(args.save_path, 'language_adapter'))
        if enabled_lora['abstractor']:
            model.abstractor.save_pretrained(os.path.join(args.save_path, 'abstractor_adapter'))
        if enabled_lora['vision']:
            model.vision_model.save_pretrained(os.path.join(args.save_path, 'vision_adapter'))
        save_lora_metadata(args.save_path, enabled_lora, args)
    else:
        model.save_pretrained(args.save_path)
        if any(enabled_lora.values()):
            save_lora_metadata(args.save_path, enabled_lora, args)


if __name__ == '__main__':
    main()
