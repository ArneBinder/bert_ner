import argparse

from data_load import ConllDataset, get_logger
from model import KerasModel, format_metrics, counts_to_metrics

logger = get_logger(__name__)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--n_epochs", type=int, default=30)
    #parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--explainable", dest="explainable", action="store_true")
    parser.add_argument("--top_rnns", dest="top_rnns", action="store_true")
    #parser.add_argument("--logdir", type=str, default="checkpoints/01")
    parser.add_argument("--trainset", type=str, default="conll2003/train.txt")
    parser.add_argument("--validset", type=str, default="conll2003/valid.txt")
    parser.add_argument("--testset", type=str, default="")
    parser.add_argument("--use_default_tagset", dest="use_default_tagset", action="store_true")
    parser.add_argument("--predict_tag", dest="predict_tag", type=str, default='ner')
    args = parser.parse_args()

    # mapping from tag type to position [column] index in the dataset
    tag_types = {'ner': 3, 'pos': 1}
    assert args.predict_tag in tag_types, \
        f'the tag type to predict [{args.predict_tag}] is not in tag_types that are taken from the datasets: ' \
        f'{", ".join(tag_types.keys())}'
    logger.info('Loading data...')
    eval_dataset = ConllDataset(args.validset, tag_types=tag_types)
    train_dataset = ConllDataset(args.trainset, tag_types=tag_types)
    maxlen = max(train_dataset.maxlen, eval_dataset.maxlen)
    train_dataset.maxlen = maxlen
    eval_dataset.maxlen = maxlen
    #tagset = eval_dataset.tagset
    logger.info('encode tokens with BERT...')
    # get bert encoded input already here to have the embedding shape for model construction
    x_train_encoded = train_dataset.x_bertencoded(keep=True)
    x_eval_encoded = eval_dataset.x_bertencoded(keep=True)
    assert x_train_encoded.shape[1:] == x_eval_encoded.shape[1:], 'shape mismatch for bert encoded sequences'
    bert_output_shape = x_train_encoded.shape[1:]
    bert_output_dtype = x_train_encoded.dtype

    default_tagsets = {'ner': ('O', 'I-LOC', 'B-PER', 'I-PER', 'I-ORG', 'I-MISC', 'B-MISC', 'B-LOC', 'B-ORG'),
                       }
    tagset = eval_dataset.generate_y(tag_type=args.predict_tag,
                                     tagset=('<PAD>',) + default_tagsets[args.predict_tag] if args.use_default_tagset else None,
                                     to_categorical=True)
    train_dataset.generate_y(tag_type=args.predict_tag, tagset=tagset, to_categorical=True)

    # TODO: put original pytorch model on top to recreate original performance
    logger.info('Build model...')
    model_keras = KerasModel(n_classes=len(tagset), n_dims=bert_output_shape[-1], lr=args.lr,
                             top_rnns=args.top_rnns)

    logger.info('Train with batch_size=%i...' % args.batch_size)
    train_metrics = model_keras.fit(train_dataset,
                      batch_size=args.batch_size,
                      n_epochs=args.n_epochs,
                      eval_dataset=eval_dataset,
                      )

    for _data, _metrics in train_metrics.items():
        logger.info(format_metrics(metrics=_metrics, prefix=_data))
        final_metrics = counts_to_metrics(**_metrics)
        logger.info(format_metrics(metrics=final_metrics, prefix=_data))

    if args.testset != '':
        logger.info('Test...')
        test_dataset = ConllDataset(args.testset, tag_types=tag_types)
        test_dataset.generate_y(tag_type=args.predict_tag, tagset=tagset, to_categorical=True)
        test_metrics = model_keras.evaluate(test_dataset=test_dataset, batch_size=args.batch_size)
        logger.info(format_metrics(metrics=test_metrics, prefix='test'))
        test_metrics_final = counts_to_metrics(**test_metrics)
        logger.info(format_metrics(metrics=test_metrics_final, prefix='test'))

    #logger.info('done')