# Decoder Only Translation Knowledge Distillation with LLaMa 3.2
The scripts in this repository demonstrate how to build a pipeline to distill the translation knowledge of LLaMa3.2 3B variant into a blank LLaMa3.2 1B config using LoRaa nd KL Divergence as part of a subclassed Trainer with additional TrainingArguments of alpha (distillation strength) and temperature (smoothing coefficient). The project builds on previous work by Lewis Tunstall explained in the book <a href="https://www.oreilly.com/library/view/natural-language-processing/9781098136789/">Natural Language Processing with Transformers</a>.

Since the distilled model starts from a blank config, its behavior is sometimes unpredictable. Namely, once translation is complete the model tends to continue generating indefinitely. To remedy this, we add a special token, <|end_of_translation|>, to the LLaMa3.2 tokenizer. This establishes a boundary so that only the initial translated text is taken into account -- the string can merely be split on this character.

Using the provided scripts, it is possible to achieve 82 COMET, which is in the high quality range. BLEU is slightly lower due to stylistic differences.

Any form of quantization will result in severe performance degradation and is not advised.

Be sure to install the following dependencies:
- transformers
- peft
- datasets
- evaluate
- tqdm
- torch
