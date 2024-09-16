from utils.utils import *
import transformers
from statistics import mean
import copy
from transformers import (
    #BertConfig,
    BertTokenizer,
    glue_output_modes,
    glue_processors
)

def string_from_id(vocab, id):
    return list(vocab.keys())[list(vocab.values()).index(id)]

def random_select(cfg, vocab):
    # choose a random token ID if a list of IDs was not provided
    if not cfg['lst_id']:
        vocab_size = len(vocab.keys())
        lst_id = sample(range(vocab_size), cfg['n_candidates'])
    else:
        lst_id = cfg['lst_id']
    ret_tokens = [string_from_id(vocab, id) for id in lst_id]
    return ret_tokens

def execute_eval(cfg, model, dataset, ret_loss=False, printing=True, all_patience=False, ret_confidence=False, ret_metrics=False):
    if 'pabee' in cfg['model_type']:
        # todo: output something to indicate progress.
        #  tqdm is not helpful because it is for every batch and I use batch size 1
        # If using PABEE, compute for each patience value of interest
        acc = []
        exit = []
        corr = []
        loss = []
        confidences = []
        if all_patience:
            patience_vals = cfg['patience_lst']
        else:
            # todo: make this not hard-coded (so that you can support models with not 12 hidden layers)
            patience_vals = [12]
        for p in patience_vals:
            result, avg_exit, all_exits, n_correct, avg_loss, lst_confidence, _ = evaluate(cfg, model, p, dataset)
            exit.append(avg_exit)
            corr.append(n_correct)
            loss.append(avg_loss)
            confidences.append(lst_confidence)
            if 'acc' in result:
                acc.append(result['acc'])

        if printing:
            print('Number of samples tested:', len(dataset))
            print('Numbers correct:', corr)
            print('Patience values tested:', cfg['patience_lst'])
            print('Accuracies:', acc)
            print('Average losses:', loss)
            print('Exit layers:', exit)

    else:
        raise NotImplementedError()
    if ret_metrics:
        return corr, exit
    if ret_loss:
        return loss
    if ret_confidence:
        return confidences

def eval_with_trigger(batch, trigger, printing=False, all_patience=False, ret_confidence=False, ret_metrics=False):
    for b in batch:
        b.text_a += ' ' + trigger
    batch_set = convert_examples_to_dataset(batch, tokenizer)
    print('*** Trying trigger:', trigger, '***')
    if ret_confidence:
        confidences = execute_eval(cfg, model, batch_set, printing=printing, all_patience=all_patience, ret_confidence=True)
        return confidences
    elif ret_metrics:
        corr, exit = execute_eval(cfg, model, batch_set, printing=printing, all_patience=all_patience, ret_metrics=True)
        return corr, exit
    else:
        loss = execute_eval(cfg, model, batch_set, ret_loss=True, printing=printing, all_patience=all_patience)
    loss = mean(loss)
    return loss

# Prepare task
processor = glue_processors[cfg['task']]()
output_mode = glue_output_modes[cfg['task']]
label_list = processor.get_labels()

# Load tokenizer and model
config_class, model_class, tokenizer_class = MODEL_CLASSES[cfg['model_type']]
config = config_class.from_pretrained(
    cfg['model_name_or_path'],
    num_labels=len(label_list),
    finetuning_task=cfg['task'],
    cache_dir=None,
)
if os.path.exists(cfg['model_name_or_path']):
    model = BertForSequenceClassificationWithPabee.from_pretrained(cfg['model_name_or_path'])
    tokenizer = BertTokenizer.from_pretrained(cfg['model_name_or_path'], do_lower_case=cfg['do_lower_case'])
# todo: can you not load the custom and pre-trained models the same way?
else:
    tokenizer = tokenizer_class.from_pretrained(
        cfg['model_name_or_path'],
        do_lower_case=cfg['do_lower_case'],
        cache_dir=None
    )
    model = model_class.from_pretrained(
        cfg['model_name_or_path'],
        from_tf=False,
        config=config,
        cache_dir=None
    )
model.to(cfg['device'])
cfg['n_layers'] = model.config.num_hidden_layers

if isinstance(model, transformers.models.albert.modeling_albert.AlbertPreTrainedModel):
    embedding_weight = model.albert.embeddings.word_embeddings.weight
elif isinstance(model, transformers.models.bert.modeling_bert.BertPreTrainedModel):
    embedding_weight = model.bert.embeddings.word_embeddings.weight
elif isinstance(model, transformers.models.deberta.modeling_deberta.DebertaPreTrainedModel):
    embedding_weight = model.deberta.embeddings.word_embeddings.weight
elif isinstance(model, transformers.models.roberta.modeling_roberta.RobertaPreTrainedModel):
    embedding_weight = model.roberta.embeddings.word_embeddings.weight
else:
    embedding_weight = model.embeddings.word_embeddings.weight

# Prepare data
all_train_examples = load_and_clean_examples_from_tsv(processor, cfg['train_dir'])
all_test_examples = load_and_clean_examples_from_tsv(processor, cfg['test_dir'])
# Create full datasets
full_train_dataset = convert_examples_to_dataset(all_train_examples, tokenizer)
full_test_dataset = convert_examples_to_dataset(all_test_examples, tokenizer)

# Create target dataset (examples with the label that the trigger will aim to cause slowdown for)
target_examples = [ex for ex in all_test_examples if ex.label == cfg['target_lbl_str']]
target_dataset = convert_examples_to_dataset(target_examples, tokenizer)
print('Found', len(target_examples), 'examples corresponding to target label', cfg['target_lbl_str'])
# todo: make it possible to get vocab from somewhere else (e.g. define a second tokenizer)
vocab = tokenizer.vocab
print('A vocabulary of size', len(vocab.keys()), 'was accessed from the loaded tokenizer')

# Initialize the trigger
# token index for 'the' is 1996
trigger = 'the'
best_loss = -1
n = cfg['perturb_batch_size']
batches = [target_examples[i * n:(i + 1) * n] for i in range((len(target_examples) + n - 1) // n)]

for a in range(cfg['n_attempts']):
    for batch_orig in batches:

        # look for a new trigger
        # candidates will be a list of candidate tokens
        candidates = random_select(cfg, vocab)

        for cnt, trigger_candidate in enumerate(candidates):
            batch = copy.deepcopy(batch_orig)
            # get performance with current trigger
            if cfg['attack_loss'] == 'new-sloth':
                loss = -1 * eval_with_trigger(batch, trigger_candidate, printing=False)
            elif cfg['attack_loss'] == 'sloth':
                loss = -1 * eval_with_trigger(batch, trigger_candidate, printing=False)
            else:
                loss = eval_with_trigger(batch, trigger_candidate, printing=False)
            if (best_loss == -1) or (loss > best_loss):
                best_loss = loss
                trigger = trigger_candidate