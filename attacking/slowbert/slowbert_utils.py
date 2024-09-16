from utils.utils import *
import csv
import copy
from torch.utils.data import DataLoader
import math
import numpy as np
import nltk
nltk.download('punkt')
import torch
import tensorflow as tf
from statistics import mean
import string
import random
from torch.nn.functional import binary_cross_entropy
from transformers import glue_compute_metrics

def evaluate(cfg, model, patience, eval_dataset, ret_lbls=False):

    if 'albert' in cfg['model_type']:
        model.albert.set_regression_threshold(cfg['regression_threshold'])
        model.albert.set_patience(patience)
        model.albert.reset_stats()
    elif 'bert' in cfg['model_type']:
        model.bert.set_regression_threshold(cfg['regression_threshold'])
        model.bert.set_patience(patience)
        model.bert.reset_stats()
    else:
        # todo: add roberta and potentially others, might just generally need to fix this conditional
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
    all_lbls = []
    correct_inds = []

    for b_ind, batch in enumerate(eval_dataloader):
        model.eval()
        batch = tuple(t.to(cfg['device']) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[3],
                      'token_type_ids': batch[2]}
            # todo: right now, only bert pabee model is returning exit layer, need to fix others
            outputs, exit_layer, layer_logits = model(**inputs)
            if cfg['task'] == 'mnli':
              logit_list = [[l1.cpu().numpy()[0][0], l1.cpu().numpy()[0][1], l1.cpu().numpy()[0][2]] for l1 in layer_logits]
            else:
              logit_list = [[l1.cpu().numpy()[0][0], l1.cpu().numpy()[0][1]] for l1 in layer_logits]
            all_layer_logits.append(logit_list)
            tmp_eval_loss, logits = outputs
            """if b_ind < 10:
              print('logits', logits, 'layer logits', layer_logits[-1:])"""
            probability = tf.nn.softmax(logits.cpu().numpy())
            # todo: do not hard code the index (so you can have more than 2 classes)
            # need to fix this, it's only returning one element of the logits
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
        """if not np.array(torch.argmax(logits.cpu())) == out_label_ids[b_ind]:
          print('INCORRECT', b_ind)"""
        """if not np.array(torch.argmax(logits.cpu())) == out_label_ids[b_ind]:
          print(out_label_ids[b_ind])"""
        # todo: this is only taking into account the incorrect/incorrect exits
        if not np.array(torch.argmax(logits.cpu())) == out_label_ids[b_ind]:
          all_exits.append(exit_layer)
          all_lbls.append(out_label_ids[b_ind])
          correct_inds.append(b_ind)
    if cfg['output_mode'] == 'classification':
        preds = np.argmax(preds, axis=1)
    else:
        # todo: add regression?
        raise NotImplementedError()
    res = glue_compute_metrics(cfg['metric_task'], preds, out_label_ids)
    results.update(res)

    if ret_lbls:
      return results, mean(all_exits), all_exits, n_correct, mean(all_loss), confidence, all_layer_logits, all_lbls
    else:
      return results, mean(all_exits), all_exits, n_correct, mean(all_loss), confidence, all_layer_logits, correct_inds

def rank_simple(data_lst):
    #return sorted(range(len(data_lst)), key=data_lst.__getitem__)
    import scipy.stats as ss
    return ss.rankdata(data_lst)

def get_slowbert_score(examples, model, p=5, a1=100):
  # I got rid of the sum vs average mode becauase that should have no effect here
  _, _, all_exits, _, _, _, all_logits, all_lbls = evaluate(cfg, model, p, examples, ret_lbls=True)

  scores = []
  for j,el in enumerate(all_logits):
    # for each example in the deletion dataset
    big_p = all_exits[j]
    # last layer logits
    final_logits = el[-1:]
    if not isinstance(final_logits, list):
      print('type problem!')
      pdb.set_trace()
    if len(final_logits) == 1:
      fp = tf.nn.softmax(final_logits[0])
    elif len(final_logits) == 2:
      fp = tf.nn.softmax(final_logits)
    else:
      print('problem!')
      pdb.set_trace()
    uniform_dist = torch.tensor([0.33333333333,0.33333333333,0.33333333333]).to(cfg["device"])
    fp_val = fp.numpy().tolist()
    fp_tensor = torch.tensor(fp_val).to(cfg["device"])
    score1 = (a1*big_p - binary_cross_entropy(fp_tensor,uniform_dist)).detach().cpu().numpy()
    # the one with the largest score is most vulnerable and should be chosen
    scores.append(-score1)
  return scores

def add_random_space(word):
    if len(word) <= 1:
        return word

    random_index = random.randint(1, len(word) - 1)
    modified_string = word[:random_index] + ' ' + word[random_index:]

    return modified_string

def swap_random_characters(input_string):
    if len(input_string) <= 1:
        return input_string

    # Choose two distinct random indices
    index1, index2 = random.sample(range(len(input_string)), 2)

    # Convert string to list to perform the swap
    modified_list = list(input_string)
    modified_list[index1], modified_list[index2] = modified_list[index2], modified_list[index1]

    modified_string = ''.join(modified_list)
    return modified_string

def add_blanks_combinations(word):
    combinations = [word]

    for i in range(len(word) - 1):
        combination = word[:i + 1] + ' ' + word[i + 1:]
        combinations.append(combination)

    return combinations

def character_swaps_combinations(word):
    combinations = []

    for i in range(len(word)):
        for j in range(i + 1, len(word)):
            swapped_chars = list(word)
            swapped_chars[i], swapped_chars[j] = swapped_chars[j], swapped_chars[i]
            combinations.append(''.join(swapped_chars))

    return combinations

def add_random_character(word):
    combinations = []

    for i in range(len(word) + 1):
        random_char = random.choice(string.ascii_lowercase)  # You can customize the set of characters if needed
        combination = word[:i] + random_char + word[i:]
        combinations.append(combination)

    return combinations

def delete_random_character(word):
    combinations = []

    for i in range(len(word)):
        combination = word[:i] + word[i + 1:]
        combinations.append(combination)

    return combinations

def generate_potential_sentences(potential_words, words, ind, tokenizer):

    # tokenize the words to compare the token count after
    length_before = len(tokenizer.tokenize(words[ind]))
    token_counts_before = np.array([length_before]*len(potential_words))
    token_counts_after = np.array([])
    for word in potential_words:
      length = len(tokenizer.tokenize(word))
      token_counts_after = np.append(token_counts_after, length)
    token_difference = token_counts_after - token_counts_before
    # generate potential sentences
    len_text = len(words)
    replaced_sentences = []
    for word in potential_words:
        replaced_sentences.append(words[0:ind] + [word] + words[ind + 1:])
    # list of sentences
    print(replaced_sentences)
    return replaced_sentences, token_difference

def are_all_same(arr):
  first_elem = arr[0]
  if ((np.array(arr) == np.array([12]*len(arr))).all()):
    return 12
  if(len(arr)==1):
    return -1

  for i in range(1,len(arr)):
    if arr[i] != first_elem:
      return 0
  return 1

def generate_slowbert(examples, model, tokenizer, output_mode, label_list, p1=5, pct_attempts=1):
  final_sentences = []
  final_labels = []

  # create masked examples
  for cnt1, ex1 in enumerate(examples):
    if cnt1 % 5 == 0:
      print('example number', cnt1)
    # get initial exit
    # convert candidates to dataset
    init_dataset = convert_examples_to_dataset([ex1], tokenizer, output_mode, label_list)
    # evaluate to get new exits
    _, _, og_exit, _, _, _, _ = evaluate(cfg, model, p1, init_dataset, ret_lbls=False)

    with_deletion = []
    tokens = tokenizer.tokenize(ex1.text_a)
    used_inds = []
    used_tokens = [] # this should ultimately just equal words (below)
    words = nltk.word_tokenize(ex1.text_a)
    words = [w for w in words if w.isalpha()]
    for i,t1 in enumerate(tokens):
      if t1 in words:
        used_inds.append(i)
        used_tokens.append(t1)
        #og_text = ex1.text_a
        #new_text = og_text.replace(str(t1), '')
        new_text_tokens = tokens.copy()
        new_text_tokens[i] = ''
        new_text = " ".join(new_text_tokens)
        new_ex = copy.deepcopy(ex1)
        new_ex.text_a = new_text
        with_deletion.append(new_ex)

    # sanity check
    """if (len(with_deletion) != len(words)) or (len(used_tokens) != len(words)):
      print('PROBLEM A')
      pdb.set_trace()"""

    # convert the new samples to a dataset
    deletion_dataset = convert_examples_to_dataset(with_deletion, tokenizer, output_mode, label_list)

    # get H scores and rank tokens accordingly
    h_scores = get_slowbert_score(deletion_dataset, model, p=p1)
    rank_scores = rank_simple(h_scores)
    ordered_inds = sorted(range(len(rank_scores)), key=rank_scores.__getitem__)

    # sanity check
    """if len(used_tokens) != len(ordered_inds):
      print('PROBLEM B')
      pdb.set_trace()"""

    if pct_attempts > 0:
      max_attempts = 1#math.ceil(len(used_tokens) * .2)
    curr_ex = copy.deepcopy(ex1)
    token_copy = tokens.copy()
    done = False
    attempts = 0
    for ind in ordered_inds:
      if not done:
        #print('CURR', curr_ex)
        potential_swaps = []
        potential_swaps = potential_swaps + add_blanks_combinations(used_tokens[ind]) #Random Blank
        potential_swaps = potential_swaps + character_swaps_combinations(used_tokens[ind]) #Random Swap
        potential_swaps = potential_swaps + add_random_character(used_tokens[ind]) #Insert Random Char
        potential_swaps = potential_swaps + delete_random_character(used_tokens[ind]) #Delete Random Char

        cand_examples = []
        for cand in potential_swaps[1:]:
          # should you even skip the first though?
          new_tokens = token_copy.copy()
          new_tokens[used_inds[ind]] = cand
          #new_cand_text = " ".join(new_tokens)
          new_cand_text = tokenizer.convert_tokens_to_string(new_tokens)
          cand_ex = copy.deepcopy(curr_ex)
          cand_ex.text_a = new_cand_text
          cand_examples.append(cand_ex)

        # convert candidates to dataset
        cand_dataset = convert_examples_to_dataset(cand_examples, tokenizer, output_mode, label_list)
        # evaluate to get new exits
        _, _, all_exits, _, _, _, _ = evaluate(cfg, model, p1, cand_dataset, ret_lbls=False)
        # choose the example with the largest exit
        exit_arr = np.array(all_exits)
        #print(all_exits)
        max_exit = max(all_exits)
        #print(max_exit)
        if max_exit == 12:
          done = True
        attempts += 1
        # stopping condition if only allowing a certain percentage of attempts
        if (pct_attempts > 0) and (attempts == max_attempts):
          done = True
        best_ind = all_exits.index(max_exit)
        best_cand = potential_swaps[1:][best_ind]
        token_copy[used_inds[ind]] = best_cand
        #curr_new_text = " ".join(token_copy)
        curr_new_text = tokenizer.convert_tokens_to_string(token_copy)
        curr_ex.text_a = curr_new_text
        """print(curr_ex)
        print('*')
        print('*')
    print(curr_ex)
    print(max_exit, '(prev', og_exit[0], ')')"""
    final_sentences.append(curr_ex.text_a)
    final_labels.append(curr_ex.label)
  return final_sentences, final_labels

def generate_slowbert_mrpc(examples, model, tokenizer, output_mode, label_list, p1=5, pct_attempts=-1):
  final_sentences = []
  final_labels = []

  # create masked examples
  for cnt1, ex1 in enumerate(examples):
    if cnt1 % 2 == 0:
      print('example number', cnt1)
    # get initial exit
    # convert candidates to dataset
    init_dataset = convert_examples_to_dataset([ex1], tokenizer, output_mode, label_list)
    # evaluate to get new exits
    _, _, og_exit, _, _, _, _ = evaluate(cfg, model, p1, init_dataset, ret_lbls=False)

    with_deletion = []
    tokens_a = tokenizer.tokenize(ex1.text_a)
    tokens_b = tokenizer.tokenize(ex1.text_b)
    token_split_ind = len(tokens_a)
    tokens = tokens_a + tokens_b
    used_inds = []
    used_tokens = [] # this should ultimately just equal words (below)
    words_a = nltk.word_tokenize(ex1.text_a)
    words_b = nltk.word_tokenize(ex1.text_b)
    words_a = [w for w in words_a if w.isalpha()]
    words_b = [w for w in words_b if w.isalpha()]
    word_split_ind = len(words_a)
    words = words_a + words_b
    for i,t1 in enumerate(tokens):
      if t1 in words:
        used_inds.append(i)
        used_tokens.append(t1)
        #og_text = ex1.text_a
        #new_text = og_text.replace(str(t1), '')
        new_text_tokens = tokens.copy()
        new_text_tokens[i] = ''
        new_text_tokens_a = new_text_tokens[:token_split_ind]
        new_text_tokens_b = new_text_tokens[token_split_ind:]
        new_text_a = " ".join(new_text_tokens_a)
        new_text_b = " ".join(new_text_tokens_b)
        new_ex = copy.deepcopy(ex1)
        new_ex.text_a = new_text_a
        new_ex.text_a = new_text_b
        with_deletion.append(new_ex)

    #pdb.set_trace()
    # sanity check
    """if (len(with_deletion) != len(words)) or (len(used_tokens) != len(words)):
      print('PROBLEM A')
      pdb.set_trace()"""

    # convert the new samples to a dataset
    deletion_dataset = convert_examples_to_dataset(with_deletion, tokenizer, output_mode, label_list)

    # get H scores and rank tokens accordingly
    h_scores = get_slowbert_score(deletion_dataset, model, p=p1)
    rank_scores = rank_simple(h_scores)
    ordered_inds = sorted(range(len(rank_scores)), key=rank_scores.__getitem__)

    # sanity check
    """if len(used_tokens) != len(ordered_inds):
      print('PROBLEM B')
      pdb.set_trace()"""

    if pct_attempts > 0:
      max_attempts = math.ceil(len(used_tokens) * .2)
    curr_ex = copy.deepcopy(ex1)
    token_copy = tokens.copy()
    done = False
    attempts = 0
    for ind in ordered_inds:
      if not done:
        #print('CURR', curr_ex)
        potential_swaps = []
        potential_swaps = potential_swaps + add_blanks_combinations(used_tokens[ind]) #Random Blank
        potential_swaps = potential_swaps + character_swaps_combinations(used_tokens[ind]) #Random Swap
        potential_swaps = potential_swaps + add_random_character(used_tokens[ind]) #Insert Random Char
        potential_swaps = potential_swaps + delete_random_character(used_tokens[ind]) #Delete Random Char

        cand_examples = []
        for cand in potential_swaps[1:]:
          # should you even skip the first though?
          new_tokens = token_copy.copy()
          new_tokens[used_inds[ind]] = cand
          new_tokens_a = new_tokens[:token_split_ind]
          new_tokens_b = new_tokens[token_split_ind:]
          #new_cand_text = " ".join(new_tokens)
          new_cand_text_a = tokenizer.convert_tokens_to_string(new_tokens_a)
          new_cand_text_b = tokenizer.convert_tokens_to_string(new_tokens_b)
          cand_ex = copy.deepcopy(curr_ex)
          cand_ex.text_a = new_cand_text_a
          cand_ex.text_b = new_cand_text_b
          cand_examples.append(cand_ex)

        # convert candidates to dataset
        cand_dataset = convert_examples_to_dataset(cand_examples, tokenizer, output_mode, label_list)
        # evaluate to get new exits
        _, _, all_exits, _, _, _, _ = evaluate(cfg, model, p1, cand_dataset, ret_lbls=False)
        # choose the example with the largest exit
        exit_arr = np.array(all_exits)
        #print(all_exits)
        max_exit = max(all_exits)
        #print(max_exit)
        if max_exit == 12:
          done = True
        attempts += 1
        # stopping condition if only allowing a certain percentage of attempts
        if (pct_attempts > 0) and (attempts == max_attempts):
          done = True
        best_ind = all_exits.index(max_exit)
        best_cand = potential_swaps[1:][best_ind]
        token_copy[used_inds[ind]] = best_cand
        #curr_new_text = " ".join(token_copy)
        token_copy_a = token_copy[:token_split_ind]
        token_copy_b = token_copy[token_split_ind:]
        curr_new_text_a = tokenizer.convert_tokens_to_string(token_copy_a)
        curr_new_text_b = tokenizer.convert_tokens_to_string(token_copy_b)
        curr_ex.text_a = curr_new_text_a
        curr_ex.text_b = curr_new_text_b
        """print(curr_ex)
        print('*')
        print('*')
    print(curr_ex)
    print(max_exit, '(prev', og_exit[0], ')')"""
    final_sentences.append([curr_ex.text_a, curr_ex.text_b])
    final_labels.append(curr_ex.label)
  return final_sentences, final_labels

def generate_slowbert_mnli(examples, model, tokenizer, output_mode, label_list, p1=5, pct_attempts=-1):
  final_sentences = []
  final_labels = []

  # create masked examples
  for cnt1, ex1 in enumerate(examples):
    if cnt1 % 2 == 0:
      print('example number', cnt1)
    # get initial exit
    # convert candidates to dataset
    init_dataset = convert_examples_to_dataset([ex1], tokenizer, output_mode, label_list)
    # evaluate to get new exits
    _, _, og_exit, _, _, _, _ = evaluate(cfg, model, p1, init_dataset, ret_lbls=False)

    with_deletion = []
    tokens_a = tokenizer.tokenize(ex1.text_a)
    tokens_b = tokenizer.tokenize(ex1.text_b)
    token_split_ind = len(tokens_a)
    tokens = tokens_a + tokens_b
    used_inds = []
    used_tokens = [] # this should ultimately just equal words (below)
    words_a = nltk.word_tokenize(ex1.text_a)
    words_b = nltk.word_tokenize(ex1.text_b)
    words_a = [w for w in words_a if w.isalpha()]
    words_b = [w for w in words_b if w.isalpha()]
    word_split_ind = len(words_a)
    words = words_a + words_b
    for i,t1 in enumerate(tokens):
      if t1 in words:
        used_inds.append(i)
        used_tokens.append(t1)
        #og_text = ex1.text_a
        #new_text = og_text.replace(str(t1), '')
        new_text_tokens = tokens.copy()
        new_text_tokens[i] = ''
        new_text_tokens_a = new_text_tokens[:token_split_ind]
        new_text_tokens_b = new_text_tokens[token_split_ind:]
        new_text_a = " ".join(new_text_tokens_a)
        new_text_b = " ".join(new_text_tokens_b)
        new_ex = copy.deepcopy(ex1)
        new_ex.text_a = new_text_a
        new_ex.text_a = new_text_b
        with_deletion.append(new_ex)

    #pdb.set_trace()
    # sanity check
    """if (len(with_deletion) != len(words)) or (len(used_tokens) != len(words)):
      print('PROBLEM A')
      pdb.set_trace()"""

    # convert the new samples to a dataset
    deletion_dataset = convert_examples_to_dataset(with_deletion, tokenizer, output_mode, label_list)
    #print('11111')
    # get H scores and rank tokens accordingly
    h_scores = get_slowbert_score(deletion_dataset, model, p=p1)
    #print('3333')
    rank_scores = rank_simple(h_scores)
    #print('4444')
    ordered_inds = sorted(range(len(rank_scores)), key=rank_scores.__getitem__)
    #print('22222')
    # sanity check
    """if len(used_tokens) != len(ordered_inds):
      print('PROBLEM B')
      pdb.set_trace()"""

    if pct_attempts > 0:
      max_attempts = math.ceil(len(used_tokens) * .2)
    curr_ex = copy.deepcopy(ex1)
    token_copy = tokens.copy()
    done = False
    attempts = 0
    for ind in ordered_inds:
      if not done:
        #print('CURR', curr_ex)
        potential_swaps = []
        potential_swaps = potential_swaps + add_blanks_combinations(used_tokens[ind]) #Random Blank
        potential_swaps = potential_swaps + character_swaps_combinations(used_tokens[ind]) #Random Swap
        potential_swaps = potential_swaps + add_random_character(used_tokens[ind]) #Insert Random Char
        potential_swaps = potential_swaps + delete_random_character(used_tokens[ind]) #Delete Random Char

        cand_examples = []
        for cand in potential_swaps[1:]:
          # should you even skip the first though?
          new_tokens = token_copy.copy()
          new_tokens[used_inds[ind]] = cand
          new_tokens_a = new_tokens[:token_split_ind]
          new_tokens_b = new_tokens[token_split_ind:]
          #new_cand_text = " ".join(new_tokens)
          new_cand_text_a = tokenizer.convert_tokens_to_string(new_tokens_a)
          new_cand_text_b = tokenizer.convert_tokens_to_string(new_tokens_b)
          cand_ex = copy.deepcopy(curr_ex)
          cand_ex.text_a = new_cand_text_a
          cand_ex.text_b = new_cand_text_b
          cand_examples.append(cand_ex)

        # convert candidates to dataset
        #print('3333')
        cand_dataset = convert_examples_to_dataset(cand_examples, tokenizer, output_mode, label_list)
        # evaluate to get new exits
        #print('4444')
        _, _, all_exits, _, _, _, _ = evaluate(cfg, model, p1, cand_dataset, ret_lbls=False)
        # choose the example with the largest exit
        exit_arr = np.array(all_exits)
        #print(all_exits)
        max_exit = max(all_exits)
        #print(max_exit)
        if max_exit == 12:
          done = True
        attempts += 1
        # stopping condition if only allowing a certain percentage of attempts
        if (pct_attempts > 0) and (attempts == max_attempts):
          done = True
        best_ind = all_exits.index(max_exit)
        best_cand = potential_swaps[1:][best_ind]
        token_copy[used_inds[ind]] = best_cand
        #curr_new_text = " ".join(token_copy)
        token_copy_a = token_copy[:token_split_ind]
        token_copy_b = token_copy[token_split_ind:]
        curr_new_text_a = tokenizer.convert_tokens_to_string(token_copy_a)
        curr_new_text_b = tokenizer.convert_tokens_to_string(token_copy_b)
        curr_ex.text_a = curr_new_text_a
        curr_ex.text_b = curr_new_text_b
        """print(curr_ex)
        print('*')
        print('*')
    print(curr_ex)
    print(max_exit, '(prev', og_exit[0], ')')"""
    final_sentences.append([curr_ex.text_a, curr_ex.text_b])
    final_labels.append(curr_ex.label)
  return final_sentences, final_labels

def write_sst(sents, lbls, dest='/content/drive/My Drive/uat/glue_data/pilot-data/sst-2/slowdown-attack/slowbert-attacked/one/dev.tsv'):
  with open(dest, 'a', newline='') as desttsv:
    fieldnames = ['sentence', 'label']
    writer = csv.DictWriter(desttsv, fieldnames=fieldnames, delimiter='\t')
    writer.writeheader()
    for i, in1 in enumerate(sents):
      writer.writerow({'sentence': in1, 'label': lbls[i]})

def write_mrpc(sents, lbls, dest='/content/drive/My Drive/uat/glue_data/pilot-data/mrpc/slowbert-attacks/slowbert-attacked/dev.tsv'):
  with open(dest, 'a', newline='') as desttsv:
        fieldnames = ['Quality', '#1 ID', '#2 ID', '#1 String', '#2 String']
        writer = csv.DictWriter(desttsv, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        for i, in1 in enumerate(sents):
            writer.writerow({'Quality': lbls[i],
                             '#1 ID': 0,
                            '#2 ID': 0,
                            '#1 String': in1[0],
                            '#2 String': in1[1]})

def write_mnli(sents, lbls, dest='/content/drive/My Drive/uat/glue_data/MNLI/100-attack-slowbert/attacked/dev_matched.tsv'):
  with open(dest, 'a', newline='') as desttsv:
        fieldnames=['index', 'promptID', 'pairID', 'genre', 'sentence1_binary_parse',
                                                      'sentence2_binary_parse', 'sentence1_parse', 'sentence2_parse',
                                                      'sentence1', 'sentence2',
                                                      'label1', 'label2', 'label3', 'label4', 'label5', 'gold_label']
        writer = csv.DictWriter(desttsv, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        for i, in1 in enumerate(sents):
            writer.writerow({'index': 0, 'promptID': 0,
                             'pairID': 0, 'genre': 0,
                             'sentence1_binary_parse': 0,
                             'sentence2_binary_parse': 0,
                             'sentence1_parse': 0, 'sentence2_parse': 0,
                             'sentence1': in1[0], 'sentence2': in1[1],
                             'label1': lbls[i], 'label2': lbls[i],
                             'label3': lbls[i], 'label4': lbls[i],
                             'label5': lbls[i], 'gold_label': lbls[i]})
