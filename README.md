# DPS: Discrepancy-Guided Parameter Suppression for Robust Fine-tuning

Code for the paper Discrepancy-Guided Parameter Suppression for Robust Fine-tuning (NeuIPS 2024 @ FITML)


## Setting up conda env
```bash
bash install.sh
mkdir checkpoints && export PYTHONPATH="$PYTHONPATH:$PWD"
```

## Datasets 

```bash
mkdir -p ./datasets/data/ && wget https://worksheets.codalab.org/rest/bundles/0x6313da2b204647e79a14b468131fcd64/contents/blob/ -O ./datasets/data/iwildcam_v2.0.zip --no-check-certificate

unzip ./datasets/data/iwildcam_v2.0.zip -d ./datasets/data/ # Try manually unzip if it doesn't work

python datacreation_scripts/iwildcam.py
```

## Launch Script

- Fine-tuning CLIP ViT-B/16 with cross entropy loss
```bash

python src/main.py --algorithm=ce --train-dataset=IWildCamIDVal --epochs=20 --lr=1e-5 --wd=0.2 --batch-size=64 --model=ViT-B/16 --eval-datasets=IWildCamIDVal,IWildCamID,IWildCamOOD --template=iwildcam_template  --save=./checkpoints/ --data-location=./datasets/data/ --ft_data="./datasets/csv/iwildcam_v2.0/train.csv" --csv-img-key filepath --csv-caption-key title --exp_name=iwildcam/ce_loss 

```

- Fine-tuning CLIP ViT-B/16 with DPS (Loaded FT model with CE loss)
```bash
python src/main.py --algorithm=dps --masking=0.9 --train-dataset=IWildCamIDVal --epochs=20 --lr=1e-5 --wd=0.2 --batch-size=64 --model=ViT-B/16 --eval-datasets=IWildCamIDVal,IWildCamID,IWildCamOOD --template=iwildcam_template  --save=./checkpoints/ --data-location=./datasets/data/ --ft_data="./datasets/csv/iwildcam_v2.0/train.csv" --csv-img-key filepath --csv-caption-key title --exp_name=iwildcam/dps_loss_mask0.9 --clip_load="./checkpoints/iwildcam/ce_loss/_BS64_WD0.2_LR1e-05_run1/checkpoint_7.pt"

```

- Fine-tuning CLIP ViT-B/16 with flyp loss
```bash
python src/main.py --algorithm=flyp --train-dataset=IWildCamIDVal --epochs=20 --lr=1e-5 --wd=0.2 --batch-size=64 --model=ViT-B/16 --eval-datasets=IWildCamIDVal,IWildCamID,IWildCamOOD --template=iwildcam_template  --save=./checkpoints/ --data-location=./datasets/data/ --ft_data="./datasets/csv/iwildcam_v2.0/train.csv" --csv-img-key filepath --csv-caption-key title --exp_name=iwildcam/flyp_loss

```


## Acknowledgments
In this code we refer to the following implementations: [FLYP](https://github.com/locuslab/FLYP/tree/main). Our code is largely built upon their wonderful implementation. 

## Reference

If our work or code helps you, please consider to cite our paper. Thank you!
```BibTeX
@inproceedings{liu2024dps,
  author = {Chang Liu, Jingyu Ma},
  title = {Discrepancy-Guided Parameter Suppression for Robust Fine-tuning},
  booktitle = {NeuIPS},
  year = {2024}
}
```