import torch
from PIL import Image
from peft import PeftModel

from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl.tokenization_mplug_owl import MplugOwlTokenizer
from mplug_owl.processing_mplug_owl import MplugOwlProcessor
from sconf import Config
from pipeline.data_utils.processors.builder import build_processors


def _load_adapter(base_module, adapter_path, adapter_name='default', is_trainable=False):
    if adapter_path is None:
        return base_module

    # First adapter wraps base module; subsequent adapters can be hot-loaded
    if isinstance(base_module, PeftModel):
        base_module.load_adapter(adapter_path, adapter_name=adapter_name, is_trainable=is_trainable)
        base_module.set_adapter(adapter_name)
        return base_module

    return PeftModel.from_pretrained(
        base_module,
        adapter_path,
        adapter_name=adapter_name,
        is_trainable=is_trainable,
    )


def get_model(
    pretrained_ckpt,
    use_bf16=False,
    language_adapter_path=None,
    abstractor_adapter_path=None,
    vision_adapter_path=None,
    adapter_name='default',
):
    """Model Provider with tokenizer and processor.

    Args:
        pretrained_ckpt (string): The path to pre-trained checkpoint.
        use_bf16 (bool, optional): Whether to use bfloat16 to load the model.
        language_adapter_path (str, optional): LoRA adapter path for language_model.
        abstractor_adapter_path (str, optional): LoRA adapter path for abstractor.
        vision_adapter_path (str, optional): LoRA adapter path for vision_model.
        adapter_name (str, optional): Adapter name for runtime switch.

    Returns:
        model: MplugOwl Model
        tokenizer: MplugOwl text tokenizer
        processor: MplugOwl processor (including text and image)
    """
    model = MplugOwlForConditionalGeneration.from_pretrained(
        pretrained_ckpt,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.half,
    )

    model.language_model = _load_adapter(model.language_model, language_adapter_path, adapter_name=adapter_name)
    model.abstractor = _load_adapter(model.abstractor, abstractor_adapter_path, adapter_name=adapter_name)
    model.vision_model = _load_adapter(model.vision_model, vision_adapter_path, adapter_name=adapter_name)

    config = Config('configs/sft/release.yaml')
    image_processor = build_processors(config['valid_processors'])['sft']
    tokenizer = MplugOwlTokenizer.from_pretrained(pretrained_ckpt)
    processor = MplugOwlProcessor(image_processor, tokenizer)
    return model, tokenizer, processor


def switch_adapter(model, adapter_name='default'):
    """Switch currently active LoRA adapter across pluggable modules."""
    for module_name in ['language_model', 'abstractor', 'vision_model']:
        module = getattr(model, module_name, None)
        if isinstance(module, PeftModel):
            module.set_adapter(adapter_name)


def unload_adapter(model):
    """Merge and unload adapters for deployment if modules are PEFT-wrapped."""
    for module_name in ['language_model', 'abstractor', 'vision_model']:
        module = getattr(model, module_name, None)
        if isinstance(module, PeftModel):
            setattr(model, module_name, module.merge_and_unload())


def do_generate(prompts, image_list, model, tokenizer, processor, use_bf16=False, **generate_kwargs):
    """The interface for generation."""
    if image_list:
        images = [Image.open(_) for _ in image_list]
    else:
        images = None
    inputs = processor(text=prompts, images=images, return_tensors='pt')
    inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        res = model.generate(**inputs, **generate_kwargs)
    sentence = tokenizer.decode(res.tolist()[0], skip_special_tokens=True)
    return sentence


if __name__ == '__main__':
    pass
