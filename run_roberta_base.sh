DATA_DIR=/private/home/zeyuliu/masking_strategy/fairseq-roberta/data-bin/wikitext-103
# OC_CAUSE=1 
# HYDRA_FULL_ERROR=1 

PYTHONPATH=. fairseq-hydra-train -m --config-dir examples/roberta/config/pretraining \
--config-name base.yaml task.data=$DATA_DIR