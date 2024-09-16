from utils.utils import *
from attacking.slowbert.slowbert_utils import *
from transformers import BertTokenizer, glue_output_modes, glue_processors

# prepare task
processor = glue_processors[cfg['task']]()
output_mode = glue_output_modes[cfg['task']]
label_list = processor.get_labels()

# load pre-fine-tuned model
checkpoint = cfg['model_path']
model = BertForSequenceClassificationWithPabee.from_pretrained(checkpoint)
tokenizer = BertTokenizer.from_pretrained(checkpoint, do_lower_case=cfg['do_lower_case'])
model.to(cfg['device'])

fname = '/content/drive/My Drive/uat/glue_data/MNLI/test/matched'
all_test_examples = load_and_clean_examples_from_tsv(processor, fname)[:100]

sents1, lbls1 = generate_slowbert_mnli(all_test_examples, model, tokenizer, output_mode, label_list)

write_mnli(sents1, lbls1)
