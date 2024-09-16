from utils.utils import *
from utils.exit_classification_utils import *
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    BertTokenizer,
    glue_output_modes,
    glue_processors,
)
from attacking.exit_classification.ExitClassifier import ExitModel

# Prepare task
processor = glue_processors[cfg['task']]()

if cfg['new_training']:
    # load tokenizer and initialize model
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=False)
    model = ExitModel()
    model.to(torch.device(cfg['device']))

    # freeze BERT layers
    lst_trainable = ['linear1.weight', 'linear1.bias',
                     'linear2.weight', 'linear2.bias',
                     'linear3.weight', 'linear3.bias']
    for name, param in model.named_parameters():
        if name not in lst_trainable:
            param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

if cfg['new_training']:
    # load training data
    if cfg['task'] == 'sst-2':
        new_train_examples = load_and_clean_examples_from_tsv(processor, file_dict['sst_train_final'][:-8])
        new_test_examples = load_and_clean_examples_from_tsv(processor, file_dict['sst_test_final'][:-8])
    elif cfg['task'] == 'mrpc':
        new_train_examples = load_and_clean_examples_from_tsv(processor, file_dict['mrpc_train_final'][:-8])
        new_test_examples = load_and_clean_examples_from_tsv(processor, file_dict['mrpc_test_final'][:-8])
    elif cfg['task'] == 'mnli':
        # treat MNLI data as MRPC data because not it is 2-sentence input with binary label
        cfg['task'] = 'sst-2'
        cfg['metric_task'] = 'sst-2'
        # Re-prepare task
        processor = glue_processors[cfg['task']]()
        output_mode = glue_output_modes[cfg['task']]
        label_list = processor.get_labels()
        new_train_examples = load_and_clean_examples_from_tsv(processor, file_dict['mnli_train_final'][:-8])
        print(len(new_train_examples))
        new_test_examples = load_and_clean_examples_from_tsv(processor, file_dict['mnli_test_final'][:-8])
        print(len(new_test_examples))
    new_train_dataset = convert_examples_to_dataset(new_train_examples, tokenizer, output_mode, label_list)
    new_test_dataset = convert_examples_to_dataset(new_test_examples, tokenizer, output_mode, label_list)

if cfg['new_training']:
    # train
    cfg['train_batch_size'] = 1024
    cfg['num_train_epochs'] = 3000
    cfg['model_name_or_path'] = ''
    global_step, tr_loss, loss_lst, train_auc_lst, eval_auc_lst = trainv3(cfg, new_train_dataset, model, tokenizer,
                                                                          optimizer, criterion, new_test_dataset)
    print("global_step = %s, average loss = %s", global_step, tr_loss)

eval_auc = evalv3(model, new_test_dataset)
print('TEST AUC:', eval_auc)

train_auc = evalv3(model, new_train_dataset)
print('TRAIN AUC:', train_auc)
