# 2nd place solution to "Google AI4Code – Understand Code in Python Notebooks" challenge

![](https://www.kaggle.com/competitions/AI4Code/discussion/343659)

The solution is based on the bi-encoder or poly-encoder like approach, when every cell is individually processed by the codebert, with the code/md interaction modelled with the pair of parallel TransformerDecoderLayer:


![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F743064%2Fc42b70aafb899ed52cf2ef7ed554e4f7%2Fcode_bert.png?generation=1660273106105753&alt=media)

I either used the shared CodeBERT based encoder for both code and markdown cells or CodeBERT for code and multi language bert for markdown. The resulting activations are averaged over tokens.

The code cell activations are passed to 1D convolution with zero padding, to convert from the N code cell positions to N+1 positions between cells (bins to put mb cell to), with every point combined activations of the cell before and after.

After adding the pos encoding to the code activations (the combination of the absolute and relative encodings).

The next stage is a few layers (2 or 6) of transformer decoders, for code cell to use the self attention and attention to md cells as a memory, the same for markdown: self attention and the memory query over the output of the code cell:
 

    for step in range(self.num_decoder_layers):
            x_code = self.code_decoders[step](x_code, x_md)
            x_md = self.md_decoders[step](x_md, x_code)


Output and loss
---------------

Outputs of the code and markdown decoders are combined with matmul similarly to attention in a transformer,
with 3 outputs:

* (nb_code+1) * nb_markdown  shape, is markdown cell i before the code bin j

* (nb_code+1) * nb_markdown  shape, is markdown cell i at the code bin j

* nb_markdown * nb_markdown is the markdown cell i before another markdown cell j

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F743064%2F863249a74f079b19943a7f6a46689e8e%2Fmodel_outputs.png?generation=1668982985055751&alt=media)

The first two outputs are used to put the md cell to the right bin between the code cells, and the last output is used to sort md cells within the single bin. BCE loss was used during training.

Post-processing
---------------

MD cell is put to the right bin between code cells using the first prediction: probability of the MD cell i is before code bin j. The position is selected to minimize the sum of probabilities of the wrong placement over all bins:


    md_cell_bin_num = np.argmin([
            -1 * md_after_code[md_idx, :i+1].sum() + md_after_code[md_idx, i+1:].sum()
            for i in range(nb_bins)
    ])

Such approach to select the markdown cell position worked much better compared to the simple "is markdown cell i at the code bin j" output + softmax/argmax. The intuition behind the loss and sorting approach was to have a proxy for expected number of cell swaps used in the competition metric.

The model was trained end to end, with the batch equal to a single notebook, I guess this makes the optimisation task harder due to the variable size of notebooks. Gradient clipping helped.

The loss weighting and notebooks sampling also had a significant impact on the score. Large notebooks have a larger impact on the final score (due to the large number of position inversions for wrong predictions) but the wrong prediction error is averaged over a large number of nb_md * nb_code predictions. To address this I sampled larger notebooks more often (most useful) and for some models added avg_loss + a * max_loss + b * sum_loss (with a very small a of 0.02, adding sum_loss helped for initial experiments but was disabled later).

Non-English notebooks handling
------------------------------

I tried to address the multiple language notebooks, I used either the same shared codebert model for code and md cells or the codebert for code and sentence-transformers/paraphrase-multilingual-mpnet-base-v2 for md.

Example of the performance difference between approaches (from the local validation, on one of folds):
shared codebert: All notebooks: 0.9113; English notebooks: 0.9164; Non-English notebooks: 0.8652
codebert+mpnet: All notebooks: 0.9088; English notebooks: 0.9117; Non-English notebooks: 0.8825

The multilingual sentence ensembling mpnet model is trained on language pairs and allows to reduce the score gap between English and non-English notebooks but it has the worse combined score (most notebooks are English).

I initially planned to do a different ensemble based on the language, but later I found the impact of ensembling codebert2 and codebert+mpnet models produces a much bigger score jump and decided to use the same large ensemble for all notebooks, regardless of the language.

Ensembling
----------

To increase the ensemble side, I also trained the separate encoder models (using per-cell predictions from the first level transformers). Such encoder only parts of models were much faster to train (single per notebook transformer compared to multiple ber cell transformers, allowed to faster experiment with different architectures) and to predict (L1 per cell model predictions are re-used).

The ensemble - to average 3 outputs between 4 folds and two models, including two versions of L2 encoders.
For ensembling I used the simple weighted averaging of model predictions (weighting was not very important, used higher weight for better performing models but the result was almost the same as for unweighted average).

External data
------------=

Models have been trained on the original dataset.
I tried to use the published external set of notebooks with acceptable licenses https://zenodo.org/record/6383115, it seems to improve results. The dataset was mentioned closer to the end of the challenge so I did not have enough time to re-train the full models. I trained the additional encoders using the external dataset notebooks, the result of individual improved, but the improvement of the final ensemble seems to be quite small.

Time permitting, I would re-train models using the external data.

The dataset converted to the challenge format: https://www.kaggle.com/datasets/dmytropoplavskiy/the-dataset-of-properlylicensed-jupyter-notebooks


What did not work or had no significant impact
----------------------------------------------

* T5 based models performed worse

* Different pre-processing of code and markdown cells had little impact, so used unprocessed.

* Different ways to combine per cell model outputs performed in a similar way (using average, avg+max, output of the special code token).

* Using GRU/LSTM instead of transformer, for the encoder part or GRU for code with transformer encoder for md/code. The performance was worse compared to the dual branch transformer decoder.


Training models
---------------

Please refer to ai4code/train_all.sh for instructions

