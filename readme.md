LLM Prompts:
- previous method: fine-tuning, 微调整个pre-trained model parameters去适应下游任务。
- 现在是prompt-tuning了，保pre-trained transformer参数不变，只训练学习额外增加的prompt embedding parameters。
  - previous prompt-tuning：hard prompt，提供一个提示词模板，微调整个预训练模型的参数。
  - current prompt-tuning 按时间顺序包括：prefix-tuning、p-tuning v1、parameter-efficient prompt tuning、p-tuning v2、Adapter、LoRA等
