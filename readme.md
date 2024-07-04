LLM Prompts:
- previous method: fine-tuning, 微调整个pre-trained model parameters去适应下游任务。
- 现在是prompt-tuning了，保pre-trained transformer参数不变，只训练学习额外增加的prompt embedding parameters。
  - previous prompt-tuning：hard prompt，构造一个提示词模板，微调整个预训练模型的参数。
    - prompt情感分类，https://blog.csdn.net/cjw838982809/article/details/12513450 -》 没啥意思，out-of-dated！abandon
  - current prompt-tuning 按时间顺序包括：prefix-tuning、p-tuning v1、parameter-efficient prompt tuning、p-tuning v2、Adapter、LoRA等。
    - LoRA 复现 01: MiniLoRA, 简单、通俗、易懂、powerful，https://github.com/cccntu/minLoRA/blob/main/demo.ipynb
    - LoRA 复现 02: LoRA from Scratch on Scratch, https://github.com/sunildkumar/lora_from_scratch/blob/main/lora_on_mnist.ipynb
    - LoRA 复现 03: PyTorch Tutorial with torchtune, https://pytorch.org/torchtune/stable/tutorials/lora_finetune.html; https://github.com/pytorch/torchtune/blob/48626d19d2108f92c749411fbd5f0ff140023a25/recipes/lora_finetune.py
