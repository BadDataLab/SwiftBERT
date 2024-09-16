from utils.utils import *
from utils.exit_classification_utils import *
from early_exit_models.pabee.modeling_pabee_bert import *
from transformers import (
    BertTokenizer,
    glue_output_modes,
    glue_processors,
)

# Prepare task
processor = glue_processors[cfg['task']]()
output_mode = glue_output_modes[cfg['task']]
label_list = processor.get_labels()

# Load tokenizer and model
config_class, model_class, tokenizer_class = MODEL_CLASSES[cfg['model_type']]
config = config_class.from_pretrained(
    cfg['model_path'],
    num_labels=len(label_list),
    finetuning_task=cfg['task'],
    cache_dir=None,
)

if os.path.exists(cfg['model_path']):
    # Loading a (pretrained) model from a provided file path
    model = BertForSequenceClassificationWithPabee.from_pretrained(cfg['model_path'])
    tokenizer = BertTokenizer.from_pretrained(cfg['model_path'], do_lower_case=cfg['do_lower_case'])
else:
    # Loading a (pretrained) model from Hugging Face, provided the model name
    # Hugging Face models can be found here: https://huggingface.co/models?sort=downloads
    tokenizer = tokenizer_class.from_pretrained(
        cfg['model_name'],
        do_lower_case=cfg['do_lower_case'],
        cache_dir=None
    )
    model = model_class.from_pretrained(
        cfg['model_name'],
        from_tf=False,
        config=config,
        cache_dir=None
    )

model.to(cfg['device'])
cfg['n_layers'] = model.config.num_hidden_layers

if cfg['make_exit_histogram']:

    # Prepare data examples
    if cfg['task'] == 'sst-2':
        train_examples = load_and_clean_examples_from_tsv(processor, file_dict['sst_train_source'])
        test_examples = load_and_clean_examples_from_tsv(processor, file_dict['sst_test_source'])
    elif cfg['task'] == 'mrpc':
        train_examples = load_and_clean_examples_from_tsv(processor, file_dict['mrpc_train_source'])
        test_examples = load_and_clean_examples_from_tsv(processor, file_dict['mrpc_test_source'])
    elif cfg['task'] == 'mnli':
        train_examples = load_and_clean_examples_from_tsv(processor, file_dict['mnli_train_source'])
        test_examples = load_and_clean_examples_from_tsv(processor, file_dict['mnli_test_source'])
    # Create full datasets with the prepared examples
    train_dataset = convert_examples_to_dataset(train_examples, tokenizer, output_mode, label_list)
    test_dataset = convert_examples_to_dataset(test_examples, tokenizer, output_mode, label_list)

    # run evaluation, while keeping record of the exits
    _, _, all_test_exits, _, _, _, _ = evaluate(cfg, model, cfg['p'], test_dataset)
    _, _, all_train_exits, _, _, _, _ = evaluate(cfg, model, cfg['p'], train_dataset)

    # make a histogram of the exits
    exit_histogram(all_train_exits, all_test_exits)

if cfg['find_confidences']:

    # Prepare data examples
    if cfg['task'] == 'sst-2':
        train_examples = load_and_clean_examples_from_tsv(processor, file_dict['sst_train_source'])
        test_examples = load_and_clean_examples_from_tsv(processor, file_dict['sst_test_source'])
    elif cfg['task'] == 'mrpc':
        train_examples = load_and_clean_examples_from_tsv(processor, file_dict['mrpc_train_source'])
        test_examples = load_and_clean_examples_from_tsv(processor, file_dict['mrpc_test_source'])
    # Create full datasets with the prepared examples
    train_dataset = convert_examples_to_dataset(train_examples, tokenizer, output_mode, label_list)
    test_dataset = convert_examples_to_dataset(test_examples, tokenizer, output_mode, label_list)

    # run evaluation, while keeping record of the performance metric, exits, and layerwise confidence scores
    test_res, _, all_test_exits, _, _, _, test_confidences = evaluate(cfg, model, cfg['p'], test_dataset)
    train_res, _, all_train_exits, _, _, _, train_confidences = evaluate(cfg, model, cfg['p'], train_dataset)

    # write confidence scores to tsvs
    if cfg['task'] == 'sst-2':
        sst2_write_to_tsv_logits(file_dict['sst_test_source'], file_dict['sst_test_unsplit'], test_confidences)
        sst2_write_to_tsv_logits(file_dict['sst_train_source'], file_dict['sst_train_unsplit'], train_confidences)
    elif cfg['task'] == 'mrpc':
        mrpc_write_to_tsv_logits(file_dict['mrpc_test_source'], file_dict['mrpc_test_unsplit'], test_confidences)
        mrpc_write_to_tsv_logits(file_dict['mrpc_train_source'], file_dict['mrpc_train_unsplit'], train_confidences)
    elif cfg['task'] == 'mnli':
        mnli_write_to_tsv_logits(file_dict['mnli_test_source'], file_dict['mnli_test_unsplit'], train_confidences)
        mnli_write_to_tsv_logits(file_dict['mnli_train_source'], file_dict['mnli_train_unsplit'], test_confidences)

if cfg['convert_dataset']:
    # use the layerwise confidence scores (which should be in tsv files) to convert the dataset
    if cfg['task'] == 'sst-2':
        convert_layer_logits_sst2(file_dict['sst_train_unsplit'], file_dict['sst_train_final'])
        convert_layer_logits_sst2(file_dict['sst_test_unsplit'], file_dict['sst_test_final'])
    if cfg['task'] == 'mrpc':
        convert_layer_logits_mrpc(file_dict['mrpc_train_unsplit'], file_dict['mrpc_train_final'])
        convert_layer_logits_mrpc(file_dict['mrpc_test_unsplit'], file_dict['mrpc_test_final'])
    if cfg['task'] == 'mnli':
        convert_layer_logits_mnli(file_dict['mnli_train_unsplit'], file_dict['mnli_train_final'])
        convert_layer_logits_mnli(file_dict['mnli_test_unsplit'], file_dict['mnli_test_final'])
