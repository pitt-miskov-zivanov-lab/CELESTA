# CELESTA

### (Context Extraction through LEarning with Semi-supervised multi-Task Architecture)

CELESTA (Context Extraction through LEarning with Semi-supervised multi-Task Architecture) is a framework for context classification in biomedical texts, applicable to both open-set and close-set scenarios.  

## Fine-tunning

To fine-tune models using **CELESTA**, the most straightforward approach is to run the provided Python training scripts or their corresponding bash wrappers. To fine-tune models with your own data, first format your dataset according to the **BioRECIPE** format (https://github.com/pitt-miskov-zivanov-lab/BioRECIPE). The classes for each task should be predefined in `labels_loader.py`.

You can then choose between two training modes:

- **Multi-task learning (MTL)** using `mtl_semi_train.py`
- **Single-task training** using `semi_train.py`

Note that we also include scripts for training baseline models in both semilearn and huggingface structure. The framework supports semi-supervised learning (SSL) algorithms such as FixMatch and VAT.

**Example (MTL with FixMatch):**

```
python mtl_semi_train.py --train="train.xlsx" --test="test.xlsx" --task="location" --alg="mtl_fixmatch" --output="./saved_models" --batch=8 --epoch=10
```

After training, use our evaluation script (e.g., `semi_evaluation.py`) to assess model performance and conduct further analysis.

## Citation

Difei Tang, Thomas Yu Chow Tam, Haomiao Luo, Cheryl A. Telmer, Natasa Miskov-Zivanov, “An Open-Set Semi-Supervised Multi-Task Learning Framework for Context Classification in Biomedical Texts”, bioRxiv preprint, doi: https://doi.org/10.1101/2024.07.22.604491.

## Funding

This work was funded in part by the NSF EAGER award CCF-2324742

## Support

This research was supported in part by the University of Pittsburgh Center for Research Computing, RRID:SCR_022735, through the resources provided. Specifically, this work used the H2P cluster, which is supported by NSF award number OAC-2117681. 