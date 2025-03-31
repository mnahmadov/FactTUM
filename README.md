# FactTUM

## Getting dataset
You can download the dataset [here](https://drive.google.com/drive/folders/1q0_MqBeGAp5_cBJCBf_1alYaYm14OeTk?usp=share_link). Use the `dbpedia_undirected_light.pickle` and `factkg.zip`. The DBpedia knowledge graph is only used during preprocessing for evidence retrieval, not training or evaluating.

Unzip `factkg.zip`, so that the `.pickle` files are saved in `factkg_dataset/`. Put the DBpedia `.pickle` file in `dbpedia/`. One can also change the paths and folder names in `constants.py`.

Check the [FactKG paper](https://arxiv.org/pdf/2104.06378) for more information about the dataset.

### Claim only
```cd claim_only```
We explore two approaches: prompting a large language model `LLaMA-3.2-3B-Instruct` and fine-tuning the same model using quantized LoRA with Unsloth, both utilizing claims only without any supporting evidence.

#### Prompting
We began with prompting to assess the capabilities of current language models. For this, we used the Llama-3.2-3B-Instruct model, which is accessible for free via [Cloudflare](https://www.cloudflare.com) Workers AI REST API. You can find the code and test-set results in `claim_only/prompting`.

All prompting approaches use Cloudflare to access `Llama-3.2-3B-Instruct` via API calls. Running these implementations requires an `account_id` and `cloudfare_api`. Set the `PATH_TO_RESULT_FILE` variable to specify the csv file path for results.

Prompting Implementation: 
```cli
python claim_only/prompting/llama_prompting.py
```

We conducted two iterations to evaluate the model’s consistency, as language models can produce varying responses for the same queries. The total accuracy scores on the test set confirmed relatively stable performance.

Prompting Results:

```claim_only/prompting/prompting_claim_only_v0.csv```

```claim_only/prompting/prompting_claim_only_v1.csv```

Results Evaluation:
```cli
python claim_only/prompting/get_statistics.py
```
`get_statistics.py` is used to evaluate results in CSV format. Set the `EVALUATION_FILE_PATH` variable to specify the file path for evaluation.

#### Fine-tuning LLaMA with Unsloth
[Unsloth](https://unsloth.ai) is an open-source library that optimizes and accelerates the fine-tuning process. For our project, we leveraged Unsloth with quantized LoRA to efficiently fine-tune within hardware constraints. We used the same model for prompting as we did for training: `LLaMA-3.2-3B-Instruct`.

We converted the original training set from the FactKG paper into a format compatible with the Llama model. Since `LLaMA-3.2-3B-Instruct` is an instruction-based model, we added a uniform instruction to all claims in the dataset to meet its training requirements.

Conversion Script: 
```cli
claim_only/finetuning/convert_to_llama_format.py
```

Converted Training Set:

```claim_only/finetuning/train_llama_format.csv```

We conducted three fine-tuning iterations, adjusting the following hyperparameters:
`Batch Size`, `Gradient Accumulation`, `Training Steps/Epochs`, `Warmup Steps`.

To run the notebooks, you need an API key for Weights & Biases (W&B). You can obtain a free API key from [Weights & Biases](https://wandb.ai/authorize),.

Notebooks:

```claim_only/finetuning/finetuned_v0/llama_finetuned.ipynb```

```claim_only/finetuning/finetuned_v1/llama_finetuned_claim_only_v1.ipynb```

```claim_only/finetuning/finetuned_v2/llama_finetuned_claim_only_v2.ipynb```


Results:

```claim_only/finetuning/finetuned_v0/fine_tuned_claim_only_v0.csv```

```claim_only/finetuning/finetuned_v1/finetuned_claim_only_v1.csv```

```claim_only/finetuning/finetuned_v2/fine_tuned_claim_only_v2.csv```

### With Evidence
```cd with_evidence```
#### Preprocess
```cd llm_filtering&gnn```
In with evidence part of the project, we trained the model on subset of the whole training set, while three steps for preprocesisng are needed:

**1. Getting samples**

one can run
```cli
python get_samples.py --sample_size_training sample_size
```
The sample size should be no more than the whole set size which is .... .Defalt size will be 20000. The training set is further split to 10 and validation set to 4. One can also change the number in constant.py.

We provided the sample we are using under /data.

**2. Filter the evidence by prompting**

```cli
python filter.evi.py --dataset_typ all
```
One can change `all` to only do a certain split. We use cloudflare API for access to llama-3.2-3b-instruct, but one could also run it locally.

Evidence will be stored under /filtered_evidence folder, and the prompting intermediate result in the corresponding csv file.

Running the prompting approach requires an `account_id` and `cloudfare_api` from [Cloudflare](https://www.cloudflare.com) Workers AI REST API. Set the 'account_id','cloudfare_api' variables to your account if you are using this method.

**3. Converting the subgraph to GNN format**

```cli
pytho n convert_subgraphs.py --dataset_type all --method []
```
Again, one can convert the subgrphs with all sets. Method is ["combined","direct"], default is "combined"

**4. precomputing the embeddings**

We are using pre-trained BERT model for embedding, from which we take only the output from [CLS]. We provide the precomputed embeddingd we are using under the folder ```embeddings```.


#### Training the model
Once the preprocessing is finished, one could start training the models with evidence.
```
python run_gnn.py model_name --gnn_type GATConv --subgraph_type combined --n_epochs 20 --batch_size 64
```
One could use the three GNN models by changing param gnn_type: ["GATConv", "GATv2Conv", "TransformerConv"].
Here are the specific script ran for the models in the paper (which includes the hyperparameters). Note that the first argument is simply the name of the model that will be saved, it can be set to anything.



qa-gnn-GATConv ```python run_gnn.py qa_gnn_gatconv  --subgraph_type combined --n_epochs 20 --batch_size 128 --gnn_
batch_norm --classifier_dropout 0.5 --gnn_dropout 0.1 --learning_rate 0.00007```

qa-gnn-GATv2Conv ```python run_gnn.py qa_gnn_gatv2conv  --n_gnn_layers 3 --subgraph_type combined --n_epochs 20 --batch_size 64 --gnn_batch_norm --classifier_dropout 0.5 --gnn_dropout 0.1 --learning_rate 0.00003```

qa-gnn-TransformerConv ```python run_gnn.py qa_gnn_trans  --n_gnn_layers 3 --subgraph_type combined --n_epochs 20 --batch_size 256 --gnn_batch_norm --classifier_dropout 0.5 --gnn_dropout 0.1 --learning_rate 0.0001```

### Prompting with Evidence
```cd with_evidence/prompting```

We experimented with a prompting-based approach using the same model as in claim-only methods, `LLaMA-3.2-3B-Instruct` while incorporating evidence retrieved through the filtering process described in `2. Filter the evidence by prompting.`

In this approach, the evidence is provided in the format of a test set, now including an additional column specifically for the retrieved evidence.

Test Sets with Evidence Included:

```with_evidence/prompting/test_with_evi_all.pickle```

```with_evidence/prompting/test_with_evi_small.pickle```

Running the prompting approach requires an `account_id` and `cloudfare_api` from [Cloudflare](https://www.cloudflare.com) Workers AI REST API. Set the `PATH_TO_RESULT_FILE` variable to specify the csv file path for results.

Prompting with Evidence Implementation:
```cli
python with_evidence/prompting/llama_prompting_with_evidence.py
```

Prompting with Evidence Results:

```with_evidence/prompting/prompting_with_evidence_v0.csv```
