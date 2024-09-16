from utils.utils import *
import transformers
from transformers import (
    glue_output_modes,
    glue_processors,
)

if cfg['fine-tune']:
    # Prepare task
    processor = glue_processors[cfg['task']]()
    output_mode = glue_output_modes[cfg['task']]
    label_list = processor.get_labels()

    # Load tokenizer and model
    config_class, model_class, tokenizer_class = MODEL_CLASSES[cfg['model_type']]
    config = config_class.from_pretrained(
        cfg['model_name'],
        num_labels=len(label_list),
        finetuning_task=cfg['task'],
        cache_dir=None,
    )
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

    # Prepare datasets
    all_train_examples = load_and_clean_examples_from_tsv(processor, file_dict['snli_train_source'])
    all_test_examples = load_and_clean_examples_from_tsv(processor, file_dict['snli_test_source'])
    full_train_dataset = convert_examples_to_dataset(all_train_examples, tokenizer, output_mode, label_list)
    full_test_dataset = convert_examples_to_dataset(all_test_examples, tokenizer, output_mode, label_list)

if cfg['fine-tune']:
    global_step, tr_loss, loss_lst, eval_auc_lst = train(cfg, full_train_dataset, model, tokenizer, full_test_dataset)
    print("global_step = %s, average loss = %s", global_step, tr_loss)
    res = evaluate(cfg, model, tokenizer, step=0)
    result = dict((k + "_{}".format(global_step), v) for k, v in res.items())
    print(result)

res, mean_exits, all_exits, n_correct, mean_loss, confidence, all_layer_logits = evaluate(cfg, model, 12,
                                                                                          full_test_dataset)

if cfg['fine-tune'] and cfg['plot_metrics']:
    import matplotlib.pyplot as plt

    plt.plot(range(len(loss_lst)), loss_lst, label='Training loss')
    plt.plot(range(len(eval_auc_lst)), eval_auc_lst, label='Validation AUC')
    plt.title('Training Metrics')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()
