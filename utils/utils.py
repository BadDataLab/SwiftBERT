import csv
import copy
import evaluate as ev
import json
import numpy as np
import os
import pdb
from random import sample
from statistics import mean
import sys

import tensorflow as tf

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from tqdm import tqdm, trange

import transformers
from transformers import (
    AdamW,
    AlbertConfig,
    AlbertTokenizer,
    AutoTokenizer,
    BertConfig,
    BertForPreTraining,
    BertTokenizer,
    get_linear_schedule_with_warmup,
    glue_compute_metrics,
    glue_convert_examples_to_features,
    glue_output_modes,
    glue_processors,
    RobertaConfig,
    RobertaTokenizer
)
from transformers.models.bert.modeling_bert import BertModel

from early_exit_models.pabee.modeling_pabee_albert import AlbertForSequenceClassificationWithPabee
from early_exit_models.pabee.modeling_pabee_bert import BertForSequenceClassificationWithPabee

from early_exit_models.deebert.modeling_highway_bert import DeeBertForSequenceClassification
from early_exit_models.deebert.modeling_highway_roberta import DeeRobertaForSequenceClassification

cfg = {
    'seed': 11,
    'output_dir': '/content/drive/My Drive/uat/saved_models/snli-exit-attack',
    'model_type': 'pabee-bert',
    'model_name': 'bert-base-uncased',
    'model_path': '/content/drive/My Drive/uat/saved_models/mnli/checkpoint-c',
    'task': 'mnli',
    'metric_task': 'mnli',
    'is_snli': True,
    'do_lower_case': True,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'n_layers': 12,
    'max_seq_length': 128,
    'num_trigger_tokens': 1,
    'p': 12,
    'patience_lst': [12],
    'regression_threshold': 0.01,
    'output_mode': 'classification',
    'eval_batch_size': 1,
    'train_batch_size': 8,
    'num_train_epochs': 100,
    'max_steps': 0,
    'save_steps': 1000000000000000000000000000,
    'save_epochs': [1000],
    'logging_steps': 383,
    'eval_steps': 1,
    'max_grad_norm': 1.0,
    'gradient_accumulation_steps': 1,
    'weight_decay': 0,
    'warmup_steps': 0,
    'learning_rate': 2e-5,
    'adam_epsilon': 1e-8,
    'evaluate_during_training': False,  # probably leave this as false for now, it's confusing
    'plot_metrics': False,
    'fine-tune': True,
    'make_exit_histogram': False,
    'find_confidences': False,
    'convert_dataset': False,
    'new_training': False
}

file_dict = {
    # source files
    'mnli_train_source':'/content/drive/My Drive/uat/glue_data/MNLI/train',
    'mnli_test_source':'/content/drive/My Drive/uat/glue_data/MNLI/test/matched',
    'sst_train_source': '/content/drive/My Drive/uat/glue_data/SST-2/train',
    'sst_test_source': '/content/drive/My Drive/uat/glue_data/SST-2/test',
    'mrpc_train_source': '/content/drive/My Drive/uat/glue_data/MRPC/train',
    'mrpc_test_source': '/content/drive/My Drive/uat/glue_data/MRPC/test',
    'snli_train_source': '/content/drive/My Drive/uat/glue_data/SNLI/train',
    'snli_test_source': '/content/drive/My Drive/uat/glue_data/SNLI/test',
    # sst-2
    'sst_train_unsplit': '/content/drive/My Drive/uat/glue_data/SST-2-exit/trainv2/dev-full.tsv',
    'sst_test_unsplit': '/content/drive/My Drive/uat/glue_data/SST-2-exit/testv2/dev-full.tsv',
    'sst_train_labeled': '/content/drive/My Drive/uat/glue_data/SST-2-exit/trainv2/dev-labeled.tsv',
    'sst_test_labeled': '/content/drive/My Drive/uat/glue_data/SST-2-exit/testv2/dev-labeled.tsv',
    'sst_train_final': '/content/drive/My Drive/uat/glue_data/SST-2-exit/trainv2/dev.tsv',
    'sst_test_final': '/content/drive/My Drive/uat/glue_data/SST-2-exit/testv2/dev.tsv',
    # mrpc
    'mrpc_train_unsplit': '/content/drive/My Drive/uat/glue_data/MRPC-exit/trainv2/dev-full.tsv',
    'mrpc_test_unsplit': '/content/drive/My Drive/uat/glue_data/MRPC-exit/testv2/dev-full.tsv',
    'mrpc_train_labeled': '/content/drive/My Drive/uat/glue_data/MRPC-exit/trainv2/D/dev-labeled.tsv',
    'mrpc_test_labeled': '/content/drive/My Drive/uat/glue_data/MRPC-exit/testv2/D/dev-labeled.tsv',
    'mrpc_train_final': '/content/drive/My Drive/uat/glue_data/MRPC-exit/trainv2/D/dev.tsv',
    'mrpc_test_final': '/content/drive/My Drive/uat/glue_data/MRPC-exit/testv2/D/dev.tsv',
    # mnli
    'mnli_train_unsplit': '/content/drive/My Drive/uat/glue_data/MNLI-exit/train/dev-full.tsv',
    'mnli_test_unsplit': '/content/drive/My Drive/uat/glue_data/MNLI-exit/test/dev-full.tsv',
    'mnli_train_final': '/content/drive/My Drive/uat/glue_data/MNLI-exit/train/dev.tsv',
    'mnli_test_final': '/content/drive/My Drive/uat/glue_data/MNLI-exit/test/dev.tsv',
    'mnli_train_final2': '/content/drive/My Drive/uat/glue_data/MNLI-exit/train/dev.tsv',
    'mnli_test_final2': '/content/drive/My Drive/uat/glue_data/MNLI-exit/test/dev.tsv',
    # snli
    'snli_train_unsplit': '/content/drive/My Drive/uat/glue_data/SNLI-exit/train/dev-full.tsv',
    'snli_test_unsplit': '/content/drive/My Drive/uat/glue_data/SNLI-exit/test/dev-full.tsv',
    'snli_train_labeled': '/content/drive/My Drive/uat/glue_data/SNLI-exit/train/dev-labeled.tsv',
    'snli_test_labeled': '/content/drive/My Drive/uat/glue_data/SNLI-exit/test/dev-labeled.tsv',
    'snli_train_final': '/content/drive/My Drive/uat/glue_data/SNLI-exit/train/dev.tsv',
    'snli_test_final': '/content/drive/My Drive/uat/glue_data/SNLI-exit/test/dev.tsv',
    # for attacking
    'sst_train_exit': '/content/drive/My Drive/uat/glue_data/SST-2-exit/trainv2',
    'sst_test_exit': '/content/drive/My Drive/uat/glue_data/SST-2-exit/testv2',
    'sst_train_attack': '/content/drive/My Drive/uat/glue_data/SST-2-exit/train-attack',
    'sst_test_attack': '/content/drive/My Drive/uat/glue_data/SST-2-exit/test-attack',
    'mrpc_test_attack': '/content/drive/My Drive/uat/glue_data/MRPC-exit/test-attack',
    'mrpc_train_attack': '/content/drive/My Drive/uat/glue_data/MRPC-exit/train-attack',
}

# Set seed
if cfg['seed']:
    torch.manual_seed(cfg['seed'])

MODEL_CLASSES = {
    'pabee-bert': (BertConfig, BertForSequenceClassificationWithPabee, BertTokenizer),
    'pabee-albert': (AlbertConfig, AlbertForSequenceClassificationWithPabee, AlbertTokenizer),
    'deebert': (BertConfig, DeeBertForSequenceClassification, BertTokenizer),
    'deeroberta': (RobertaConfig, DeeRobertaForSequenceClassification, RobertaTokenizer),
}


def load_and_clean_examples_from_tsv(processor, directory):
    # load dataset
    input_examples = processor.get_dev_examples(directory)
    print(
        'Loaded', len(input_examples), 'total examples from', directory + '/dev.tsv for the', cfg['task'], 'task.')

    # there will be extra \'s in the labels, so remove them
    # will end up with a list where each example is of type 'transformers.data.processors.utils.InputExample',
    # which will be of form, e.g.,
    # (guid='dev-1', text_a='hide new secretions from the parental units ', text_b=None, label='0')
    examples = []
    for ex in input_examples:
        ex.label = ex.label.replace('\'', '')
        examples.append(ex)

    return examples


def convert_examples_to_dataset(examples, tokenizer, output_mode, label_list):
    if cfg['task'] in ["mnli", "mnli-mm"] and cfg['model_type'] in ["roberta", "xlmroberta"]:
        # HACK(label indices are swapped in RoBERTa pretrained model)
        label_list[1], label_list[2] = label_list[2], label_list[1]
    features = glue_convert_examples_to_features(
        examples,
        tokenizer,
        label_list=label_list,
        max_length=cfg['max_seq_length'],
        output_mode=output_mode,
    )

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if cfg['output_mode'] == 'classification':
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    else:
        raise NotImplementedError()

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

    return dataset


def loss_to_csv(tr_loss, loss_scalar, losscsv='loss.csv'):
    with open(losscsv, 'a', newline='') as desttsv:
        fieldnames = ['loss_scalar']
        writer = csv.DictWriter(desttsv, fieldnames=fieldnames, delimiter=',')
        writer.writerow({'loss_scalar': loss_scalar})


def acc_to_csv(results, acccsv='acc.csv'):
    with open(acccsv, 'a', newline='') as desttsv:
        fieldnames = ['acc', 'f1', 'acc_and_f1']
        writer = csv.DictWriter(desttsv, fieldnames=fieldnames, delimiter=',')
        writer.writerow({'acc': results['acc'],
                         'f1': results['f1'],
                         'acc_and_f1': results['acc_and_f1']})


def train(cfg, train_dataset, model, tokenizer, eval_dataset=[]):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=cfg['train_batch_size'])

    if cfg['max_steps'] > 0:
        t_total = cfg['max_steps']
        cfg['num_train_epochs'] = cfg['max_steps'] // (len(train_dataloader) // cfg['gradient_accumulation_steps']) + 1
    else:
        t_total = len(train_dataloader) // cfg['gradient_accumulation_steps'] * cfg['num_train_epochs']

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": cfg['weight_decay'],
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=cfg['learning_rate'], eps=cfg['adam_epsilon'])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg['warmup_steps'], num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist

    # Train!
    print("***** Running training *****")
    print("  Num examples = %d", len(train_dataset))
    print("  Num Epochs = %d", cfg['num_train_epochs'])
    print("  Total train batch size (w. parallel, distributed & accumulation) = %d",
          cfg['train_batch_size'] * cfg['gradient_accumulation_steps'])
    print("  Gradient Accumulation steps = %d", cfg['gradient_accumulation_steps'])
    print("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained,
        int(cfg['num_train_epochs']),
        desc="Epoch",
        disable=False,
    )
    train_it = 1
    train_epoch = 1

    all_samples = {}

    eval_acc_lst = []
    loss_lst = []

    for _ in train_iterator:
        print('\n', 'TRAIN ITERATION', train_it, '\n')
        train_it += 1
        samples_used = set()
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)
        for step, batch in enumerate(epoch_iterator):
            samples_used.add(batch[0])
            train_epoch += 1
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(cfg['device']) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3], "token_type_ids": batch[2]}
            outputs, _, _ = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if cfg['gradient_accumulation_steps'] > 1:
                loss = loss / cfg['gradient_accumulation_steps']

            loss.backward()

            tr_loss += loss.item()

            if (step + 1) % cfg['gradient_accumulation_steps'] == 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg['max_grad_norm'])

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if cfg['logging_steps'] > 0 and global_step % cfg['logging_steps'] == 0:
                    logs = {}
                    if (cfg['evaluate_during_training'] and eval_dataset) and (global_step % cfg['eval_steps'] == 0):
                        results, _, _, _, _, _, _ = evaluate(cfg, model, cfg['n_layers'], eval_dataset)
                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value
                        acc_to_csv(results)

                    loss_scalar = (tr_loss - logging_loss) / cfg['logging_steps']
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    loss_to_csv(tr_loss, loss_scalar)

                    print(json.dumps({**logs, **{"step": global_step}}))

            if cfg['max_steps'] > 0 and global_step > cfg['max_steps']:
                epoch_iterator.close()
                print('stopping point 1', global_step, cfg['max_steps'])
                break

        loss_lst.append(tr_loss / global_step)
        print('TRAIN LOSS:', loss_lst)

        if train_it in cfg['save_epochs']:
            output_dir = os.path.join(cfg['output_dir'], "checkpoint-{}".format(global_step))
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print("Saving model checkpoint to", output_dir)
            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            print("Saving optimizer and scheduler states to", output_dir)

        if (global_step % 10 == 0):
            print("Calculating evaluation result...")
            res, mean_exits, all_exits, n_correct, mean_loss, confidence, all_layer_logits = evaluate(cfg, model, 12,
                                                                                                      eval_dataset)
            eval_acc_lst.append(res['mnli/acc'])
            print('TEST RESULTS:', res)

        if cfg['max_steps'] > 0 and global_step > cfg['max_steps']:
            train_iterator.close()
            print('stopping point 2', global_step, cfg['max_steps'])
            break

    return global_step, tr_loss / global_step, loss_lst, eval_acc_lst


def evaluate(cfg, model, patience, eval_dataset):
    if 'albert' in cfg['model_type']:
        model.albert.set_regression_threshold(cfg['regression_threshold'])
        model.albert.set_patience(patience)
        model.albert.reset_stats()
    elif 'bert' in cfg['model_type']:
        model.bert.set_regression_threshold(cfg['regression_threshold'])
        model.bert.set_patience(patience)
        model.bert.reset_stats()
    else:
        raise NotImplementedError()

    # Note that DistributedSampler samples randomly
    eval_dataloader = DataLoader(eval_dataset, batch_size=cfg['eval_batch_size'], shuffle=False)
    results = {}
    eval_loss = 0.0
    nb_eval_steps = 0
    n_correct = 0
    preds = None
    out_label_ids = None
    all_exits = []
    all_loss = []
    confidence = []
    all_layer_logits = []

    for b_ind, batch in enumerate(eval_dataloader):
        model.eval()
        batch = tuple(t.to(cfg['device']) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[3],
                      'token_type_ids': batch[2]}
            outputs, exit_layer, layer_logits = model(**inputs)
            logit_list = [[l1.cpu().numpy()[0][0], l1.cpu().numpy()[0][1]] for l1 in layer_logits]
            all_layer_logits.append(logit_list)
            tmp_eval_loss, logits = outputs
            probability = tf.nn.softmax(logits.cpu().numpy())
            confidence.append(probability.numpy()[0][1])
            eval_loss += tmp_eval_loss.mean().item()
            all_loss.append(tmp_eval_loss.mean().item())
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
        n_correct += np.array(torch.argmax(logits.cpu())) == out_label_ids[b_ind]
        all_exits.append(exit_layer)
    if cfg['output_mode'] == 'classification':
        preds = np.argmax(preds, axis=1)
    else:
        raise NotImplementedError()
    res = glue_compute_metrics(cfg['metric_task'], preds, out_label_ids)
    results.update(res)

    return results, mean(all_exits), all_exits, n_correct, mean(all_loss), confidence, all_layer_logits
