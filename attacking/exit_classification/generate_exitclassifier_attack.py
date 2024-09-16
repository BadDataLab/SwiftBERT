from utils.utils import *
from utils.exit_classification_utils import *
import torch
from transformers import (
    BertTokenizer,
    glue_output_modes,
    glue_processors,
)

# prepare task
cfg['task'] = 'sst-2'
cfg['metric_task'] = 'sst-2'
processor = glue_processors[cfg['task']]()
output_mode = glue_output_modes[cfg['task']]
label_list = processor.get_labels()
new_test_examples = load_and_clean_examples_from_tsv(processor, file_dict['mnli_test_final'][:-8])

# load pretrained model
model_cp = torch.load(cfg['output_dir'] + '/checkpoint-50/checkpoint.pth', map_location=torch.device('cpu'))
tokenizer_cp = cfg['output_dir'] + '/checkpoint-50'
model1 = model_cp['model']
optimizer = model_cp['optimizer']
tokenizer = BertTokenizer.from_pretrained(tokenizer_cp, do_lower_case=cfg['do_lower_case'])
model1.to(cfg['device'])
eval_dataset = convert_examples_to_dataset2(new_test_examples, tokenizer, output_mode, label_list)

# check eval AUC
eval_auc = evalv3(model1, eval_dataset)
print('TEST AUC:', eval_auc)
