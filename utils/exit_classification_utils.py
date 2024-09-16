import csv
import evaluate as ev
from utils.utils import *
import tensorflow as tf
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import get_linear_schedule_with_warmup, glue_convert_examples_to_features
from tqdm import tqdm, trange

def exit_histogram(train_exits, test_exits):
    import matplotlib.pyplot as plt
    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                      '#f781bf', '#a65628', '#984ea3',
                      '#999999', '#e41a1c', '#dede00']

    plt.hist(train_exits, alpha=0.5, color=CB_color_cycle[0], label='Train samples')
    plt.hist(test_exits, alpha=0.5, color=CB_color_cycle[1], label='Test samples')
    plt.ylabel('Count')
    plt.xlabel('Exit Layer')
    plt.legend()
    plt.show()


def sst2_write_to_tsv(source, dest, values):
    """Write some provided label data ('values' parameter, for example, a list
    of exits) to a file where the corresponding input will be the sentences
    provided in the 'source' file"""
    all_in = []
    with open(source + '/dev.tsv') as sourcetsv:
        reader = csv.reader(sourcetsv, delimiter='\t', quotechar='|')
        header = False
        for row in reader:
            if header:
                all_in.append(row[0])
            else:
                header = True

    with open(dest, 'a', newline='') as desttsv:
        fieldnames = ['sentence', 'label']
        writer = csv.DictWriter(desttsv, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        for i, in1 in enumerate(all_in):
            writer.writerow({'sentence': in1,
                             'label': values[i]})


def sst2_write_to_tsv_logits(source, dest, values):
    all_in = []
    with open(source + '/dev.tsv') as sourcetsv:
        reader = csv.reader(sourcetsv, delimiter='\t', quotechar='|')
        header = False
        for row in reader:
            if header:
                all_in.append(row)
            else:
                header = True

    with open(dest, 'a', newline='') as desttsv:
        fieldnames = ['sentence', 'label', 'logits']
        writer = csv.DictWriter(desttsv, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        for i, in1 in enumerate(all_in):
            writer.writerow({'sentence': in1[0],
                             'label': in1[1],
                             'logits': values[i]})


def mrpc_write_to_tsv(source, dest, values):
    all_in = []
    with open(source + '/dev.tsv') as sourcetsv:
        reader = csv.reader(sourcetsv, delimiter='\t', quotechar='|')
        header = False
        for row in reader:
            if header:
                all_in.append(row)
            else:
                header = True

    with open(dest, 'a', newline='') as desttsv:
        fieldnames = ['Quality', '#1 ID', '#2 ID', '#1 String', '#2 String']
        writer = csv.DictWriter(desttsv, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        for i, in1 in enumerate(all_in):
            writer.writerow({'Quality': values[i],
                             '#1 ID': in1[1],
                             '#2 ID': in1[2],
                             '#1 String': in1[3],
                             '#2 String': in1[4]})


def mrpc_write_to_tsv_logits(source, dest, values):
    all_in = []
    with open(source + '/dev.tsv') as sourcetsv:
        reader = csv.reader(sourcetsv, delimiter='\t', quotechar='|')
        header = False
        for row in reader:
            if header:
                all_in.append(row)
            else:
                header = True

    with open(dest, 'a', newline='') as desttsv:
        fieldnames = ['Quality', '#1 ID', '#2 ID', '#1 String', '#2 String', 'logits']
        writer = csv.DictWriter(desttsv, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        for i, in1 in enumerate(all_in):
            writer.writerow({'Quality': in1[0],
                             '#1 ID': in1[1],
                             '#2 ID': in1[2],
                             '#1 String': in1[3],
                             '#2 String': in1[4],
                             'logits': values[i]})


def mnli_write_to_tsv(source, dest, values):
    all_in = []
    with open(source + '/dev.tsv') as sourcetsv:
        reader = csv.reader(sourcetsv, delimiter='\t', quotechar='|')
        header = False
        for row in reader:
            if header:
                all_in.append(row[0])
            else:
                header = True

    with open(dest, 'a', newline='') as desttsv:
        fieldnames = ['sentence', 'label']
        writer = csv.DictWriter(desttsv, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        for i, in1 in enumerate(all_in):
            writer.writerow({'sentence': in1,
                             'label': values[i]})


def mnli_write_to_tsv_logits(source, dest, values):
    all_in = []
    with open(source + '/dev_matched.tsv') as sourcetsv:
        reader = csv.reader(sourcetsv, delimiter='\t', quotechar='|')
        header = False
        for row in reader:
            if header:
                all_in.append(row)
            else:
                header = True

    with open(dest, 'a', newline='') as desttsv:
        fieldnames = ['index', 'promptID', 'pairID', 'genre',
                      'sentence1_binary_parse', 'sentence2_binary_parse',
                      'sentence1_parse', 'sentence2_parse',
                      'sentence1', 'sentence2',
                      'gold_label', 'logits']
        writer = csv.DictWriter(desttsv, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        for i, in1 in enumerate(all_in):
            writer.writerow({'index': in1[0],
                             'promptID': in1[1],
                             'pairID': in1[2],
                             'genre': in1[3],
                             'sentence1_binary_parse': in1[4],
                             'sentence2_binary_parse': in1[5],
                             'sentence1_parse': in1[6],
                             'sentence2_parse': in1[7],
                             'sentence1': in1[8],
                             'sentence2': in1[9],
                             'gold_label': in1[len(in1) - 1],
                             'logits': values[i]})


def convert_layer_logits_sst2(source, dest, class0=[0, 1, 2, 3, 4], class1=[7, 8, 9, 10, 11, 12], threshold=.9):
    """Convert a list of logits computed at every hidden layer into a binary exit label"""
    all_in = []
    with open(source) as sourcetsv:
        reader = csv.reader(sourcetsv, delimiter='\t', quotechar='|')
        header = False
        for row in reader:
            if header:
                all_in.append(row)
            else:
                header = True
    cnt0 = 0
    cnt1 = 0
    cntnone = 0
    with open(dest, 'a', newline='') as desttsv:
        fieldnames = ['sentence', 'label']
        writer = csv.DictWriter(desttsv, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        for i, in1 in enumerate(all_in):
            logits = in1[2].split('], [')
            logits = [l1.replace('[', '').replace(']', '').split(',') for l1 in logits]
            float_logits = []
            for l in logits:
                if l is not 'logits':
                    float_logits.append([float(l[0].replace('\"', '')), float(l[1].replace('\"', ''))])
            probabilities = [tf.nn.softmax(l1) for l1 in float_logits]
            found_layer = False
            final_label = -1
            j = 0
            while not found_layer and j < len(probabilities):
                # if the label is correct at the layer
                if (np.argmax(probabilities[j]) == int(in1[1])) and max(probabilities[j]) >= threshold:
                    found_layer = j
                j += 1
            if found_layer:
                if found_layer in class0:
                    final_label = 0
                    cnt0 += 1
                elif found_layer in class1:
                    final_label = 1
                    cnt1 += 1
                else:
                    cntnone += 1
                if final_label > -1:
                    writer.writerow({'sentence': in1[0],
                                     'label': final_label})
    print('***')
    print('For', source + ':')
    print('Labeled', cnt0, 'examples with class 0')
    print('Labeled', cnt1, 'examples with class 1')
    print(cntnone, 'examples were rejected')
    print('***')


def convert_layer_logits_mrpc(source, dest, class0=[0, 1, 2, 3], class1=[5, 6, 7, 8, 9, 10, 11, 12], threshold=.9):
    all_in = []
    with open(source) as sourcetsv:
        reader = csv.reader(sourcetsv, delimiter='\t', quotechar='|')
        header = False
        for row in reader:
            if header:
                all_in.append(row)
            else:
                header = True
    cnt0 = 0
    cnt1 = 0
    cntnone = 0
    with open(dest, 'a', newline='') as desttsv:
        fieldnames = ['Quality', '#1 ID', '#2 ID', '#1 String', '#2 String']
        writer = csv.DictWriter(desttsv, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        for i, in1 in enumerate(all_in):
            logits = in1[5].split('], [')
            logits = [l1.replace('[', '').replace(']', '').split(',') for l1 in logits]
            float_logits = []
            for l in logits:
                float_logits.append([float(l[0]), float(l[1])])
            probabilities = [tf.nn.softmax(l1) for l1 in float_logits]
            found_layer = False
            final_label = -1
            j = 0
            while not found_layer and j < len(probabilities):
                # if the label is correct at the layer
                if (np.argmax(probabilities[j]) == int(in1[0])) and max(probabilities[j]) >= threshold:
                    found_layer = j
                j += 1
            if found_layer:
                if found_layer in class0:
                    final_label = 0
                    cnt0 += 1
                elif found_layer in class1:
                    final_label = 1
                    cnt1 += 1
                else:
                    cntnone += 1
                if final_label > -1:
                    writer.writerow({'Quality': final_label,
                                     '#1 ID': in1[1],
                                     '#2 ID': in1[2],
                                     '#1 String': in1[3],
                                     '#2 String': in1[4]})
    print('***')
    print('For', source + ':')
    print('Labeled', cnt0, 'examples with class 0')
    print('Labeled', cnt1, 'examples with class 1')
    print(cntnone, 'examples were rejected')
    print('***')


def convert_layer_logits_mnli(source, dest, class0=[0, 1, 2, 3], class1=[7, 8, 9, 10, 11, 12], threshold=.9):
    """Convert a list of logits computed at every hidden layer into a binary exit label"""
    all_in = []
    with open(source) as sourcetsv:
        reader = csv.reader(sourcetsv, delimiter='\t', quotechar='|')
        header = False
        for row in reader:
            if header:
                all_in.append(row)
            else:
                header = True
    cnt0 = 0
    cnt1 = 0
    cntnone = 0
    correct_labels = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
    with open(dest, 'a', newline='') as desttsv:
        fieldnames = ['index', 'promptID', 'pairID', 'genre',
                      'sentence1_binary_parse', 'sentence2_binary_parse',
                      'sentence1_parse', 'sentence2_parse',
                      'sentence1', 'sentence2',
                      'gold_label']
        writer = csv.DictWriter(desttsv, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        for i, in1 in enumerate(all_in):
            logits = in1[11].split('], [')
            logits = [l1.replace('[', '').replace(']', '').split(',') for l1 in logits]
            float_logits = []
            for l in logits:
                if l is not 'logits':
                    float_logits.append([float(l[0].replace('\"', '')), float(l[1].replace('\"', ''))])
            probabilities = [tf.nn.softmax(l1) for l1 in float_logits]
            found_layer = False
            final_label = -1
            j = 0
            while not found_layer and j < len(probabilities):
                # if the label is correct at the layer
                if (np.argmax(probabilities[j]) == correct_labels[in1[10]]) and max(probabilities[j]) >= threshold:
                    found_layer = j
                j += 1
            if found_layer:
                if found_layer in class0:
                    final_label = 0
                    cnt0 += 1
                elif found_layer in class1:
                    final_label = 1
                    cnt1 += 1
                else:
                    cntnone += 1
                if final_label > -1:
                    writer.writerow({'index': in1[0],
                                     'promptID': in1[1],
                                     'pairID': in1[2],
                                     'genre': in1[3],
                                     'sentence1_binary_parse': in1[4],
                                     'sentence2_binary_parse': in1[5],
                                     'sentence1_parse': in1[6],
                                     'sentence2_parse': in1[7],
                                     'sentence1': in1[8],
                                     'sentence2': in1[9],
                                     'gold_label': final_label})

    print('***')
    print('For', source + ':')
    print('Labeled', cnt0, 'examples with class 0')
    print('Labeled', cnt1, 'examples with class 1')
    print(cntnone, 'examples were rejected')
    print('***')


def get_text_list_from_csv_mrpc(csvfile, tsv=False):
    text_lst = []
    label_lst = []
    with open(csvfile) as f:
        if tsv:
            reader = reader = csv.reader(f, delimiter='\t', quotechar='|')
        else:
            reader = csv.reader(f, delimiter=',', quotechar='|')
        for row in reader:
            if 'Quality' not in row[0]:
                label_lst.append(int(row[0]))
                # 3 is string 1, 4 is string 2
                text_lst.append(row[3] + row[4])
    return text_lst, label_lst


def get_text_list_from_csv_sst2(csvfile, tsv=True):
    text_lst = []
    label_lst = []
    with open(csvfile) as f:
        if tsv:
            reader = reader = csv.reader(f, delimiter='\t', quotechar='|')
        else:
            reader = csv.reader(f, delimiter=',', quotechar='|')
        for row in reader:
            if 'sentence' not in row[0]:
                text_lst.append(row[0])
                label_lst.append(int(row[1]))
    return text_lst, label_lst


def prepare_text_lst(text_lst, tokenizer):
    token_tensor_lst = []
    segments_tensors_lst = []

    for text in text_lst:
        # Add the special tokens.
        marked_text = "[CLS] " + text + " [SEP]"
        # Split the sentence into tokens.
        tokenized_text = tokenizer.tokenize(marked_text)
        # Map the token strings to their vocabulary indeces.
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        # Mark each of the 22 tokens as belonging to sentence "1".
        segments_ids = [1] * len(tokenized_text)
        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        token_tensor_lst.append(tokens_tensor)
        segments_tensors_lst.append(segments_tensors)

    return token_tensor_lst, segments_tensors_lst


def get_sentence_embeddings(token_tensor_lst, segments_tensors_lst, model, layer=-2):
    embedding_lst = []

    hidden_states_lst = []
    with torch.no_grad():
        for i in range(len(token_tensor_lst)):
            outputs = model(token_tensor_lst[i], segments_tensors_lst[i])
            hidden_states = outputs[2]
            hidden_states_lst.append(hidden_states)

    for hs in hidden_states_lst:
        # token_vecs is a tensor with shape [22 x 768]
        # I think this is just looking at the penultimate layer (when layer=-2)
        token_vecs = hs[layer][0]
        # Calculate the average of all 22 token vectors.
        sentence_embedding = torch.mean(token_vecs, dim=0)
        embedding_lst.append(sentence_embedding)

    return embedding_lst


def write_embeddings_to_tsv_sst2(embeddings, label_lst, dest):
    with open(dest, 'a', newline='') as desttsv:
        fieldnames = ['sentence', 'label']
        writer = csv.DictWriter(desttsv, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        for i, in1 in enumerate(embeddings):
            writer.writerow({'sentence': in1,
                             'label': label_lst[i]})


def add_embeddings_sst2(tsvfile, destfile, model, tokenizer, tsv=True, layer=-2):
    # Preparing the input sentence(s)
    text_lst, label_lst = get_text_list_from_csv_sst2(tsvfile, tsv)
    token_tensor_lst, segments_tensors_lst = prepare_text_lst(text_lst, tokenizer)

    # Run the text through BERT, and collect all of the hidden states produced from all 12 layers.
    embeddings = get_sentence_embeddings(token_tensor_lst, segments_tensors_lst, model, layer)
    write_embeddings_to_tsv_sst2(embeddings, label_lst, destfile)


def write_embeddings_to_tsv_mrpc(embeddings, label_lst, dest):
    with open(dest, 'a', newline='') as desttsv:
        fieldnames = ['sentence', 'label']
        writer = csv.DictWriter(desttsv, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        for i, in1 in enumerate(embeddings):
            writer.writerow({'sentence': in1,
                             'label': label_lst[i]})


def add_embeddings_mrpc(tsvfile, destfile, model, tokenizer, tsv=True, layer=-2):
    # Preparing the input sentence(s)
    text_lst, label_lst = get_text_list_from_csv_mrpc(tsvfile, tsv)
    token_tensor_lst, segments_tensors_lst = prepare_text_lst(text_lst, tokenizer)

    # Run the text through BERT, and collect all of the hidden states produced from all 12 layers.
    embeddings = get_sentence_embeddings(token_tensor_lst, segments_tensors_lst, model, layer)
    write_embeddings_to_tsv_mrpc(embeddings, label_lst, destfile)


def mrpc_to_sst(source, dest):
    all_in = []
    with open(source + '/dev-mrpc.tsv') as sourcetsv:
        reader = csv.reader(sourcetsv, delimiter='\t', quotechar='|')
        header = False
        for row in reader:
            if header:
                all_in.append(row)
            else:
                header = True

    with open(dest, 'a', newline='') as desttsv:
        fieldnames = ['sentence', 'label']
        writer = csv.DictWriter(desttsv, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        for i, in1 in enumerate(all_in):
            writer.writerow({'sentence': in1[3] + ' ' + in1[4],
                             'label': in1[0]})


def trainv3(cfg, train_dataset, model, tokenizer, optimizer, criterion, eval_dataset=[]):
    # Mainly for training the exit model. Different from previous train function
    # because the model saving needs to be different, some of the training metric
    # computation is different, the returned parameters are different

    loss_lst = []
    eval_auc_lst = []
    train_auc_lst = []

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=cfg['train_batch_size'])

    if cfg['max_steps'] > 0:
        t_total = cfg['max_steps']
        cfg['num_train_epochs'] = cfg['max_steps'] // (len(train_dataloader) // cfg['gradient_accumulation_steps']) + 1
    else:
        t_total = len(train_dataloader) // cfg['gradient_accumulation_steps'] * cfg['num_train_epochs']

    # Prepare optimizer and schedule (linear warmup and decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg['warmup_steps'], num_training_steps=t_total
    )
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
    if os.path.exists(cfg['model_name_or_path']):
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(cfg['model_name_or_path'].split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) // cfg['gradient_accumulation_steps'])
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // cfg['gradient_accumulation_steps'])

        print("  Continuing training from checkpoint, will skip to saved global_step")
        print("  Continuing training from epoch %d", epochs_trained)
        print("  Continuing training from global step %d", global_step)
        print("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

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

    for _ in train_iterator:
        print('\n', 'TRAIN ITERATION', train_it, '\n')
        train_it += 1
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)
        for step, batch in enumerate(epoch_iterator):
            train_epoch += 1
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            labels = batch[3].tolist()
            ind0 = []
            ind1 = []
            for i in range(len(labels)):
                if labels[i] == 0:
                    ind0.append(i)
                else:
                    ind1.append(i)

            # print(len(ind0))
            # print(len(ind1))

            if ind0:
                id_lst0 = []
                mask_lst0 = []
                target_lst0 = []
                for i0 in ind0:
                    id_lst0.append(batch[0][i0])
                    mask_lst0.append(batch[1][i0])
                    target_lst0.append(batch[3][i0])
                # print(len(id_lst0))
                id0 = torch.stack(id_lst0).to(cfg['device'])
                mask0 = torch.stack(mask_lst0).to(cfg['device'])
                target0 = torch.stack(target_lst0).to(cfg['device'])
                inputs = {"ids": id0, "mask": mask0}
                targets = target0
                output = model(**inputs)
                output = nn.functional.log_softmax(output, dim=1)
                loss = criterion(output, targets)
                loss.backward()

            if ind1:
                id_lst1 = []
                mask_lst1 = []
                target_lst1 = []
                for i1 in ind1:
                    id_lst1.append(batch[0][i1])
                    mask_lst1.append(batch[1][i1])
                    target_lst1.append(batch[3][i1])
                id1 = torch.stack(id_lst1).to(cfg['device'])
                mask1 = torch.stack(mask_lst1).to(cfg['device'])
                target1 = torch.stack(target_lst1).to(cfg['device'])
                inputs = {"ids": id1, "mask": mask1}
                targets = target1
                output = model(**inputs)
                output = nn.functional.log_softmax(output, dim=1)
                loss = criterion(output, targets)
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
            # Save model checkpoint
            output_dir = os.path.join(cfg['output_dir'], "checkpoint-{}".format(global_step))

            print("Saving model checkpoint to", output_dir)
            torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
            tokenizer.save_pretrained(output_dir)
            print("Saving optimizer and scheduler states to", output_dir)

        if (global_step % cfg['eval_steps'] == 0):
            print("Calculating evaluation AUC...")
            eval_auc = evalv3(model, eval_dataset)
            eval_auc_lst.append(eval_auc)
            print('TEST AUC:', eval_auc_lst)

        if cfg['max_steps'] > 0 and global_step > cfg['max_steps']:
            train_iterator.close()
            print('stopping point 2', global_step, cfg['max_steps'])
            break

    return global_step, tr_loss / global_step, loss_lst, train_auc_lst, eval_auc_lst


def evalv3(model, eval_dataset, metric="roc_auc"):
    # for sst
    res_lst = []
    metric = ev.load(metric)
    model.eval()

    eval_dataloader = DataLoader(eval_dataset, batch_size=cfg['eval_batch_size'], shuffle=False)
    for b_ind, batch in enumerate(eval_dataloader):
        model.eval()
        batch = tuple(t.to(cfg['device']) for t in batch)
        with torch.no_grad():
            inputs = {"ids": batch[0], "mask": batch[1]}
            targets = batch[3]
            logits = model(**inputs)
            predictions = torch.argmax(logits, dim=-1)
            predictions = predictions.tolist()
            targets = targets.tolist()
            metric.add_batch(prediction_scores=predictions, references=targets)
            # print(predictions)
            # print(targets)
            # print('*')
            # metric.add_batch(prediction_scores=[0, 0, 0, 0, 1], references=[0, 0, 0, 1, 1])
    res_lst.append(metric.compute()['roc_auc'])
    if len(res_lst) == 1:
        return res_lst[0]
    return res_lst


def convert_examples_to_dataset2(examples, tokenizer, output_mode, label_list):
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

    for i, ft in enumerate(features):
        cfg['eval_in'][tuple(ft.input_ids)] = [i, examples[i].text_a, examples[i].text_b]

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


def record_correct(fpath, data_lst, original_src_fpath):
    all_in = []
    all_text = []
    with open(original_src_fpath + '/dev_matched.tsv') as sourcetsv:
        reader = csv.reader(sourcetsv, delimiter='\t', quotechar='|')
        header = False
        for row in reader:
            if header:
                all_in.append(row)
                concat_text = row[8] + row[9]
                all_text.append(concat_text.replace(' ', '').replace('\"', '').replace('\'', ''))
            else:
                header = True

    with open(fpath, 'a', newline='') as desttsv:
        fieldnames = ['index', 'promptID', 'pairID',
                      'sentence1_binary_parse', 'sentence2_binary_parse',
                      'sentence1_parse', 'sentence2_parse',
                      'sentence1', 'sentence2',
                      'gold_label']
        writer = csv.DictWriter(desttsv, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        for d in data_lst:
            ind = all_text.index(d)
            if ind < 0:
                pdb.set_trace()
            else:
                in1 = all_in[ind]
                writer.writerow({'index': in1[0],
                                 'promptID': in1[1],
                                 'pairID': in1[2],
                                 'sentence1_binary_parse': in1[4],
                                 'sentence2_binary_parse': in1[5],
                                 'sentence1_parse': in1[6],
                                 'sentence2_parse': in1[7],
                                 'sentence1': in1[8],
                                 'sentence2': in1[9],
                                 'gold_label': in1[15]})


def eval_and_record_correct(model, eval_dataset, original_src_fpath):
    lst0 = []
    lst1 = []
    n_corr = 0
    cnt0 = 0
    cnt1 = 0
    eval_dataloader = DataLoader(eval_dataset, batch_size=cfg['eval_batch_size'], shuffle=False)
    for b_ind, batch in enumerate(eval_dataloader):
        model.eval()
        batch = tuple(t.to(cfg['device']) for t in batch)
        with torch.no_grad():
            inputs = {"ids": batch[0], "mask": batch[1]}
            target = batch[3]
            id = tuple(batch[0][0].tolist())
            data = cfg['eval_in'][id]
            logits = model(**inputs)
            predictions = torch.argmax(logits, dim=-1)
            predictions = predictions.tolist()
            targets = target.tolist()
            if target == predictions[0]:
                n_corr += 1
                text = data[1].replace(' ', '').replace('\"', '').replace('\'', '')
                if target == 0:
                    lst0.append(text)
                    cnt0 += 1
                elif target == 1:
                    lst1.append(text)
                    cnt1 += 1
    record_correct(file_dict['mnli_exit_0_correct'], lst0, original_src_fpath)
    record_correct(file_dict['mnli_exit_1_correct'], lst1, original_src_fpath)
    print(n_corr, 'out of', len(eval_dataset), 'were correct')
    print(cnt0, 'were 0 and', cnt1, 'were 1')
    return 0
