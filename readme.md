LLM Prompts:
- previous method: fine-tuning, 微调整个pre-trained model parameters去适应下游任务。
  - previous prompt-tuning：hard prompt，构造一个提示词模板，微调整个预训练模型的参数。
    - prompt情感分类，https://blog.csdn.net/cjw838982809/article/details/12513450 -》 没啥意思，out-of-dated！abandon
- 现在是prompt-tuning了，保pre-trained transformer参数不变，只训练学习额外增加的prompt embedding parameters。
  - current prompt-tuning 按时间顺序包括：prefix-tuning、p-tuning v1、parameter-efficient prompt tuning、p-tuning v2、Adapter、LoRA等。
  - LoRA 复现 01: MiniLoRA, 简单、通俗、易懂、powerful，https://github.com/cccntu/minLoRA/blob/main/demo.ipynb
