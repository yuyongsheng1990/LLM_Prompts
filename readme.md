LLM Prompts:
- previous method: fine-tuning, 微调整个pre-trained model parameters去适应下游任务。
  - previous prompt-tuning：hard prompt，构造一个提示词模板，微调整个预训练模型的参数。
    - prompt情感分类，https://blog.csdn.net/cjw838982809/article/details/12513450 -》 没啥意思，out-of-dated！abandon
- 现在是prompt-tuning了，保pre-trained transformer参数不变，只训练学习额外增加的prompt embedding parameters。
  - current prompt-tuning 按时间顺序包括：prefix-tuning、p-tuning v1、parameter-efficient prompt tuning、p-tuning v2、Adapter、LoRA等。
  - LLM tuning推荐学习资料：
    - https://github.com/datawhalechina/self-llm/tree/master
    - https://github.com/KMnO4-zx
  - Repository: 嬛嬛chat
    - LoRA 复现 01: GLM-4-9B-chat Lora LoRA微调通用语言模型 Generative LM：https://github.com/datawhalechina/self-llm/blob/master/GLM-4/05-GLM-4-9B-chat%20Lora%20%E5%BE%AE%E8%B0%83.ipynb
    
