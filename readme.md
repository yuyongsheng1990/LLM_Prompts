LLM Prompts:
- previous method: fine-tuning, 微调整个pre-trained model parameters去适应下游任务。
- 现在是prompt-tuning了，保pre-trained transformer参数不变，只训练学习额外增加的prompt embedding parameters。
  - previous prompt-tuning：hard prompt，构造一个提示词模板，微调整个预训练模型的参数。
    - prompt情感分类，https://blog.csdn.net/cjw838982809/article/details/12513450 -》 没啥意思，out-of-dated！abandon
  - current prompt-tuning 按时间顺序包括：prefix-tuning、p-tuning v1、parameter-efficient prompt tuning、p-tuning v2、Adapter、LoRA等。
    - LoRA 复现 01: MiniLoRA, 简单、通俗、易懂、powerful，https://github.com/cccntu/minLoRA/blob/main/demo.ipynb
    - LoRA 复现 02: LoRA from Scratch on MNIST, https://github.com/sunildkumar/lora_from_scratch/blob/main/lora_on_mnist.ipynb
    - LoRA 复现 03: PyTorch Tutorial with torchtune, https://pytorch.org/torchtune/stable/tutorials/lora_finetune.html;
    - LoRA 复现 04: peft Implementation, https://github.com/hahuyhoang411/LoRA-Implementation/blob/main/prepare_data.py
    *** 选看，太难，不做实现
    - *LoRA 05: Explanation, https://blog.csdn.net/GarryWang1248/article/details/135036298
    - *LoRA 06: Huanhuan chat, https://github.com/datawhalechina/self-llm/blob/master/GLM-4/05-GLM-4-9B-chat%20Lora%20%E5%BE%AE%E8%B0%83.ipynb; https://blog.csdn.net/FL1623863129/article/details/139585668
