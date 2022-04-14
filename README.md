# Title: BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension

#### Group Members' Names: Hina Shafique Awan, Sidda De Silva


#### Group Members' Emails: hina.awan@ryerson.ca , ravindra.desilva@ryerson.ca

# Introduction:

#### Problem Description:

Self-supervised models have made significant progress in the field of Natural Language Processing. Masked Language model is the most remarkable approach in self supervised settings. However, these methods are task specific and have limited application. We need a model that can work in a wide range of end tasks. BART is a model which combines both Bidirectional and Auto-Regressive Transformers and pretrains a model. It can ben seen as generalizing many pretraining schemes and achieve noising flexibility.

#### Context of the Problem:

Emergence of pre-trained models have revolutionised the applications of natural language processing. New developers can now build applications using pre-trained models and experts can acheive better results by fine tuning the models for the downstream tasks. This saves them from the effort of training their models from the begining and focus more on building advanced applications.

Pre-trained models can be easily incorporated in new applications as they don't need much-labelled data, making it  adaptable to various business problems; prediction, transfer learning or feature extraction.

However, the most pre-trained models do not perform well across different tasks. BART instead is a generalized pre-trained model that has a wider applicability.

#### Limitation About other Approaches:

#### BERT: 
Random tokens are replaced with masks, and the document is bidirectionally encoded. BERT does not perform very well on downstream text generation tasks as the masked tokens are predicted independent of each other.

#### GPT: 
Tokens are predicted auto-regressively.  This means that each new prediction uses previously predicted tokens as context. This helps it perform very well on downstream text generation tasks. However, words can only condition on leftward context, so it cannot learn bidirectional interactions.

#### Solution:

BART combines Bidirectional and Auto-Regressive transformers in order to pre train a model. Noising flexibility is the key advantage of this approach; arbitrary transformations can be applied to the original text, including changing its length. The paper evaluates a number of noising approaches, finding the best performance by both randomly shuffling the order of the original sentences and using a novel in-filling scheme, where arbitrary length spans of text (including zero length) are replaced with a single mask token. This approach generalizes the original word masking and next sentence prediction objectives in BERT by forcing the model to reason more about overall sentence length and make longer range transformations to the input.

# Background

The related work is shown in the following table:

| Reference |Explanation |  Dataset/Input |Weakness
| --- | --- | --- | --- |
| Dong et al. [6] | UniLM is pretrained using three types of language modeling tasks: unidirectional, bidirectional, and sequence-to-sequence prediction for downstream tasks like extractive question answering, long text generation and abstractive summatization, repectively.| CNN/DailyMail dataset and Gigaword for model fine tuning and evaluation | Experimental results demonstrate that the model compares favorably with BERT on the GLUE benchmark and two question answering datasets. In future, the UNIM can be extended to support cross-lingual tasks
| Devlin et al. [5] | BERT introduced masked language modelling, which allows pre-training to learn interactions between left and right context words. Predictions are not made auto-regressively which reduces the effectiveness of BERT for generation tasks.| GLUE, MultiNLI, SQUAD v1.1, SQUAD v2.0 | GLUE score is 80.5%, MultiNLI accuracy is 86.7%, F1 score of SQUAD v1.1 is 93.2, F1 score of SQUAD v2.0 is 83.1
| Yinhan et al. [2] | RoBERTa is a robustly optimized method built on BERT architecture and modifies key hyperparameters by removing the next-sentence pretraining objective and is trained on much larger dataset.| GLUE and SQuAD | Score of 88.5 on the public GLUE leaderboard
| Lewis et al. [4] | BART is a denoising autoencoder built with a sequence-to-sequence model that is applicable in several ways for the target applications.| SQUAD for question answering, MNLI for classification, XSum for news summarization, ConvAI2 for dialogue response, ELI5 dataset for abstrative QA, CNN/Daily Mail data set for summarization | Text infilling shows strong performance.



# Methodology

In order to understand the architecture of BART, we first need to percieve the nature of the NLP tasks it aims to deal with. In text summarization and question answering tasks it is essential for our model to read the whole text and perceive each token in the context of what came before and after it. For example, if we want to traing a masked language model with the sentence “the man went to the dairy store to buy a gallon of milk”, we can use the following sentence as an input: 

“the man went to the [MASK] store to buy a gallon of milk”.

For NLU problem like this, it is important to completely read the sentence before predicting the [Mask] as it is highly dependent on the words "milk" and "store". The bi-directional approaches to read and represent a text can properly interpret input sequences in these cases. BERT (Devlin et al., 2019) introduced masked language
modelling, allowing pre-training to learn interactions between left and right context words, hence improving the task of language modeling.

BART uses the bi-directional encoder of BERT to find the best representation of its input sequence. The BERT encoder produces an embedding vector of each token of its input text sequence and an extra vector of sentence level information. This helps the decoder learn token and sentence level tasks which helps in fine-tuning of future tasks.

The masked sequences are used in the pre-training process as shown below. While simple token masking technique is used in pre training the BERT model, BART utilizes more challenging masking procedure in its pre-training.

![Alternate text ](encoder.webp "Title of the figure, location is simply the directory of the notebook")

A decoder is needed, after getting the representations of input text sequence, to map these with the target output. However, if the decoder is designed in a similar way, it can perform poorly on next sentence prediction and token prediction tasks as it depends on more diverse input.

In such cases, a model architecture is needed that can be trained on producing the next word by only examining preceding words in the sequence. Hence, an autoregressive model is useful as it only looks at the past data to predict the future. 

Below image shows how the autoregressive decoder processes its input.

![Alternate text ](auto-decoder.webp "Title of the figure, location is simply the directory of the notebook")

BART attaches the bi-directional encoder to the autoregressive decoder to create a denoising auto-encoder architecture. Based on these two components, the final BART model would look something like this:

![Alternate text ](bart-arch-main.webp "Title of the figure, location is simply the directory of the notebook")

In the above figure, the input sequence is a masked (or noisy) version of [ABCDE] transformed into [A[MASK]B[MASK]E]. The encoder examins the whole sequence and learns high-dimensional representations with bi-directional information. The autoregressive decoder takes these thought vectors and predicts the next token based on the encoder input and the output tokens predicted so far. Learning occurs by computing and optimizing the negative log-likelihood as mapped with the target [ABCDE]. With the help of BART, we can properly understand the inputs and generate new outputs.

For example, in the case of text summarization

![Alternate text ](bart-arch1.jpeg "Title of the figure, location is simply the directory of the notebook")

In a classic transformer architecture, BART is an essential component. 
![Alternate text ](transformer.webp "Title of the figure, location is simply the directory of the notebook")

## BART vs Traditional tranformer
The BART system is built on a typical sequence-to-sequence transformer. Both sections of the Transformer architecture are used in sequence-to-sequence models.
The encoder's attention layers can access all of the words in the original phrase at each step, but the decoder's attention layers can only access the words positioned before a particular word in the input.ReLU activation functions are used in traditional Sequence-to-Sequence models, although BART employs GeLUs. In the most recent Transformers, the GeLU activation function was employed in Google's BERT and OpenAI's GPT-2. 
![Alternate text ](gelu_equation.png "Title of the figure, location is simply the directory of the notebook")
![Alternate text ](gelu.png "Title of the figure, location is simply the directory of the notebook")

Then why is it considered as a performant model for multiple downstream tasks? BART borrows pre-training ideas from BERT and adds more pre-training objectives to generalize the model

## Pretraining objectives/tasks
BART takes a semi-supervised approach to learning. First, the model is pre-trained on tokens “t” looking back to “k” tokens in the past to compute the current token. This is done unsupervised on a vast text corpus to allow the model to “learn the language.”

While the model architecture is straightforward, the work's main contribution is a thorough examination of the numerous pretraining activities. While many other papers were about “oh we used this pretraining task along with others and got better performance! WOW”, this paper is more about “from all those many pretraining tasks, which are really helpful and effective?”

The goal of the pretraining exercises is to recover from document corruption. There are five different sorts of "noising" methods employed. The model will learn to generalize by using multiple tasks.

![Alternate text ](masking.png "Title of the figure, location is simply the directory of the notebook")

### Token masking
Follows BERT where some tokens are masked, and the model needs to predict the masked token.

###  Token deletion
Delete token and make the model to restore deleted token at the right position.

### Text infilling
Multiple words are selected in a span, and replaced with single MASK token. This will teach model to predict how many tokens are missing.

### Sentence permutation
Shuffle sentences and make model restore them.

### Document rotation
Select random token. Change document ordering to start from selected token. Make model predict the start of the original document.

## Pretraining experimentation
The paper's experiment on pretraining tasks may be stated as follows: 

- The effectiveness of pretraining approaches varies greatly depending on the task. 
- The importance of token masking cannot be overstated. 
- Pretraining from left to right enhances generation. 
- SQuAD requires bi-directional encoders. 
- The pre-training goal isn't the only aspect to consider. Architectural considerations such as relative position embeddings, segment-level recurrence, and so on are important. 
- On the ELI5 downstream task, pure language models perform best. 
- BART consistently delivers the best results. 

# Implementation

Unlike BERT, which only has an encoder, and GPT, which only has a decoder, Bart has both an encoder and a decoder. 
class BartModel(BartPretrainedModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)
Bart's encoder and decoder are similar to those found in any transformer model. The distinction is in the manner in which BART has been trained. 

## Fine tuning

BERT, like the other Transformer models described, is trained as a language model.  This implies that BERT is trained on enormous volumes of raw text self-supervised way. Self-supervised learning is a sort of training in which the goal is calculated automatically from the model's inputs. That implies the data doesn't need to be labelled by humans! 

This sort of model creates a statistical comprehension of the language it was trained on, but it isn't very effective for specific tasks. As a result, the general pretrained model undergoes a process known as transfer learning. The model is fine-tuned in a supervised manner — that is, using human-annotated labels — on a specific task during this process.

Predicting the next word in a phrase after reading the previous n words is an example of a task. Because the result is dependent on the past and present inputs but not the future ones, this is referred to as causal language modelling. 

Masked language modelling is another example, in which the model predicts a masked word in a phrase. 

Pretraining is the process of training a model from the ground up: the weights are set at random and the training begins with no past information. On the other hand, fine-tuning refers to the training that occurs after a model has been pre-trained. To fine-tune a language model, you must first obtain a pre-trained model, then undertake further training with a dataset unique to your goal. The question arises, Why not just train for the final task straight away? There are several factors at play: 

- The pretrained model had already been trained on a dataset that resembled the fine-tuning dataset in certain ways. As a result, the fine-tuning procedure might benefit from the knowledge gained by the original model during pretraining.  (In NLP issues, for example, the pretrained model will have some statistical grasp of the language you're working with).
- Because the pretrained model has already trained on a large amount of data, fine-tuning takes far less data to achieve acceptable results. As a result, the time and resources required to get good outcomes are significantly reduced.  For example, a science/research-based model may be created by using a pretrained model trained on the English language and then fine-tuning it on an arXiv corpus. 
- The fine-tuning will only need a little quantity of data: the information obtained by the pretrained model is "transferred," that is why it is named as transfer learning. 


# I have rephrased until here......
### Text summarization

The input sequence is fed to encoder, while the decoder autoregressively generates output.


```python
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig
import wikipedia
wikisearch = wikipedia.page("Microsoft")
wikicontent = wikisearch.content

# Loading the model and tokenizer for bart-large-cnn
tokenizer=BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model=BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

inputs = tokenizer.batch_encode_plus([wikicontent],return_tensors='pt',truncation=True)
summary_ids = model.generate(inputs['input_ids'], early_stopping=True)
summary_ids

# Decoding and printing the summary
bart_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print(bart_summary)
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    <ipython-input-1-21f188b9b9ec> in <module>
    ----> 1 from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig
          2 import wikipedia
          3 wikisearch = wikipedia.page("Microsoft")
          4 wikicontent = wikisearch.content
          5 


    ModuleNotFoundError: No module named 'transformers'


## Classification

The same input sequence is fed to encoder and decoder.
Final hidden state of final decoder token is used for classification.


```python
from transformers import pipeline
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")
sequence_to_classify = "one day I will see the world"
candidate_labels = ['travel', 'cooking', 'dancing']
classifier(sequence_to_classify, candidate_labels, multi_class=True)
```


    Downloading:   0%|          | 0.00/1.13k [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/1.52G [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/26.0 [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/878k [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/446k [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/1.29M [00:00<?, ?B/s]





    {'sequence': 'one day I will see the world',
     'labels': ['travel', 'dancing', 'cooking'],
     'scores': [0.9938650727272034, 0.003273805370554328, 0.0028610476292669773]}



## Question and answering


```python
from transformers import BartTokenizer, BartForQuestionAnswering
import torch

tokenizer = BartTokenizer.from_pretrained('a-ware/bart-squadv2')
model = BartForQuestionAnswering.from_pretrained('a-ware/bart-squadv2')

question, text = "Which name is also used to describe the Amazon rainforest in English?", "The Amazon rainforest (Portuguese: Floresta Amazônica or Amazônia; Spanish: Selva Amazónica, Amazonía or usually Amazonia; French: Forêt amazonienne; Dutch: Amazoneregenwoud), also known in English as Amazonia or the Amazon Jungle, is a moist broadleaf forest that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 square kilometres (2,700,000 sq mi), of which 5,500,000 square kilometres (2,100,000 sq mi) are covered by the rainforest. This region includes territory belonging to nine nations. The majority of the forest is contained within Brazil, with 60% of the rainforest, followed by Peru with 13%, Colombia with 10%, and with minor amounts in Venezuela, Ecuador, Bolivia, Guyana, Suriname and French Guiana. States or departments in four nations contain 'Amazonas' in their names. The Amazon represents over half of the planet's remaining rainforests, and comprises the largest and most biodiverse tract of tropical rainforest in the world, with an estimated 390 billion individual trees divided into 16,000 species."
encoding = tokenizer(question, text, return_tensors='pt')
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']

start_scores, end_scores = model(input_ids, attention_mask=attention_mask, output_attentions=False)[:2]

all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])
answer = tokenizer.convert_tokens_to_ids(answer.split())
answer = tokenizer.decode(answer)
print(answer)
```

     as Amazonia or the Amazon Jungle



```python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained("Primer/bart-squad2")
model = AutoModelForQuestionAnswering.from_pretrained("Primer/bart-squad2")
# model.to('cuda'); 
model.eval()

def answer(question, text):
    seq = '<s>' +  question + ' </s> </s> ' + text + ' </s>'
    tokens = tokenizer.encode_plus(seq, return_tensors='pt', padding='max_length', max_length=1024)
    input_ids = tokens['input_ids']
    attention_mask = tokens['attention_mask']
    start, end= model(input_ids, attention_mask=attention_mask)[:2]
    start_idx = int(start.argmax().int())
    end_idx =  int(end.argmax().int())
    print(tokenizer.decode(input_ids[0, start_idx:end_idx]).strip())
    # ^^ it will be an empty string if the model decided "unanswerable"

question = "Where does Tom live?"
context = "Tom is an engineer in San Francisco."
answer(question, context)
```

    San Francisco



```python
question, text = "Which name is also used to describe the Amazon rainforest in English?", "The Amazon rainforest (Portuguese: Floresta Amazônica or Amazônia; Spanish: Selva Amazónica, Amazonía or usually Amazonia; French: Forêt amazonienne; Dutch: Amazoneregenwoud), also known in English as Amazonia or the Amazon Jungle, is a moist broadleaf forest that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 square kilometres (2,700,000 sq mi), of which 5,500,000 square kilometres (2,100,000 sq mi) are covered by the rainforest. This region includes territory belonging to nine nations. The majority of the forest is contained within Brazil, with 60% of the rainforest, followed by Peru with 13%, Colombia with 10%, and with minor amounts in Venezuela, Ecuador, Bolivia, Guyana, Suriname and French Guiana. States or departments in four nations contain 'Amazonas' in their names. The Amazon represents over half of the planet's remaining rainforests, and comprises the largest and most biodiverse tract of tropical rainforest in the world, with an estimated 390 billion individual trees divided into 16,000 species."
answer(question, text)
```

    Amazonia or the Amazon Jungle



```python
question = "Where does Tom live?"
context = "Tom is an engineer in San Francisco."
answer(question, context)
```

    San Francisco


## Machine translation

At first glance, one may ask: isn’t this just a part of “sequence generation” task? and yes it is but this specific task takes a slightly different approach.
Instead of finetuning the BART model itself to the current downstream task(machine translation), it uses a pretrained BART model as a submodel, where another small encoder is attached to the BART encoder.
This configuration is to show that a pretrained BART model itself as a whole can be utilized by adding the small front encoder for machine translation task on a new language.
The existing BART’s first encoder’s embedding layer is replaced to a randomly initialized encoder, and then the entire model is trained end-to-end. This new encoder can use a separate vocabulary from the pretrained one.
When fine tuning in this configuration, the training is split to two phases. The first phase will only train parameters of the new encoder, BART positional embeddings, and self-attention input projection matrix of BART’s first encoder layer. On the second phase, all model parameters are updated.

# Conclusion and Future Direction

Write what you have learnt in this project. In particular, write few sentences about the results and their limitations, how they can be extended in future. Make sure your own inference/learnings are depicted here.

# References:

[1]:  Devlin, J., Chang, M., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. ArXiv, abs/1810.04805.

[2]:  Liu, Yinhan et al. “RoBERTa: A Robustly Optimized BERT Pretraining Approach.” ArXiv abs/1907.11692 (2019): n. pag.

[3]:  Yang, Zhilin et al. “XLNet: Generalized Autoregressive Pretraining for Language Understanding.” NeurIPS (2019).

[4]: Lewis, M., Liu, Y., Goyal, N., Ghazvininejad, M., Mohamed, A., Levy, O., Stoyanov, V., & Zettlemoyer, L. (2020). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension. ArXiv, abs/1910.13461.

[5]:  Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 4171–
4186, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1423. URL https://www.aclweb.org/anthology/N19-1423.

[6]:  L. Dong, N. Yang, W. Wang, F. Wei, X. Liu, Y. Wang, J. Gao, M. Zhou, and H. Hon (2019) Unified language model pre-training for natural language understanding and generation. arXiv preprint arXiv:1905.03197. 


```python

```
