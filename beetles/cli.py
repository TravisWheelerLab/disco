from argparse import ArgumentParser
from .inference_utils import DEFAULT_MODEL_DIRECTORY


def parser():
    ap = ArgumentParser()
    # infer
    subparsers = ap.add_subparsers(title='actions', required=True, dest="command")

    infer_parser = subparsers.add_parser("infer",
                                         add_help=True)
    infer_parser.add_argument('--saved_model_directory', required=False,
                              default=DEFAULT_MODEL_DIRECTORY,
                              type=str, help='where the ensemble of models is stored')
    infer_parser.add_argument('--model_extension', default='.pt', type=str,
                              help='filename extension of saved model files')
    infer_parser.add_argument('--wav_file', required=True, type=str,
                              help='.wav file to predict')
    infer_parser.add_argument('--output_csv_path', default=None, type=str,
                              help='where to save the final predictions')
    infer_parser.add_argument('--tile_overlap', default=128, type=int,
                              help='how much to overlap consecutive predictions. Larger values will mean slower '
                                   'performance as '
                                   'there is more repeated computation')
    infer_parser.add_argument('--tile_size', default=1024, type=int,
                              help='length of input spectrogram')
    infer_parser.add_argument('--batch_size', default=32, type=int,
                              help='batch size')
    infer_parser.add_argument('--input_channels', default=98, type=int,
                              help='number of channels of input spectrogram')
    infer_parser.add_argument('--hop_length', type=int, default=200,
                              help='length of hops b/t subsequent spectrogram windows')
    infer_parser.add_argument('--vertical_trim', type=int, default=30,
                              help='how many rows to remove from the spectrogram ')
    infer_parser.add_argument('--n_fft', type=int, default=800,
                              help='size of the fft to use when calculating spectrogram')
    infer_parser.add_argument('--debug', type=str, default=None,
                              help='where to save debugging data')
    infer_parser.add_argument('--num_threads', type=int, default=4,
                              help='how many threads to use when evaluating on CPU')

    # train
    train_parser = subparsers.add_parser("train", add_help=True)

    tunable = train_parser.add_argument_group(title='tunable args', description='arguments in this group are'
                                                                                'tunable, and tuned in hparam_optimizer.py')
    tunable.add_argument('--n_fft', type=int, required=True,
                         help='number of ffts used to create the spectrogram')
    tunable.add_argument('--learning_rate', type=float, required=True,
                         help='initial learning rate')
    tunable.add_argument('--vertical_trim', type=int, required=True,
                         help='how many rows to remove from the low-frequency range of the spectrogram.'
                              'This is probably unnecessary because NaNs are easily removed in preprocessing.')
    tunable.add_argument('--mask_beginning_and_end', action='store_true',
                         help='whether or not to mask the beginning and end of single-label chirps')
    tunable.add_argument('--begin_mask', type=int, default=30,
                         help='how many cols to mask from beginning')
    tunable.add_argument('--end_mask', type=int, default=10,
                         help='how many cols to mask from end')
    non_tunable = train_parser.add_argument_group(title='non-tunable args',
                                                  description='the "mel" argument depends on the data'
                                                              'extraction step - whether or not a mel'
                                                              'spectrogram was computed')
    non_tunable.add_argument('--log', action='store_true', help='whether or not to apply a log2 transform to the'
                                                                'spectrogram')
    non_tunable.add_argument('--apply_attn', action='store_true', help='use 1d Unet with attention')
    non_tunable.add_argument('--mel', action='store_true', help='use a mel-transformed spectrogram')
    non_tunable.add_argument('--bootstrap', action='store_true', help='train a model with a sample of the training set'
                                                                      '(replace=True)')
    non_tunable.add_argument('--batch_size', type=int, required=True, help='batch size')
    non_tunable.add_argument('--tune_initial_lr', action='store_true', help='whether or not to use PyTorchLightning\'s'
                                                                            'built-in initial LR tuner')
    non_tunable.add_argument('--gpus', type=int, required=True, help='number of gpus per node')
    non_tunable.add_argument('--num_nodes', type=int, required=True, help='number of nodes. If you want to train with 8'
                                                                          'GPUs, --gpus should be 4 and --num_nodes'
                                                                          'should be 2 (assuming you have 4 GPUs per '
                                                                          'node')
    non_tunable.add_argument('--epochs', type=int, required=True, help='max number of epochs to train')
    non_tunable.add_argument('--check_val_every_n_epoch', type=int, required=False,
                             default=1, help='how often to validate the model. On each validation run the loss is '
                                             'logged '
                                             'and if it\'s lower than the previous best the current model is saved')
    non_tunable.add_argument('--log_dir', type=str, required=True, help='where to save the model logs (train, test '
                                                                        'loss '
                                                                        'and hyperparameters). Visualize with '
                                                                        'tensorboard')
    non_tunable.add_argument('--data_path', type=str, required=True, help='where the data are saved on disk. Assumes'
                                                                          'the data were saved with np.save and reside'
                                                                          'in <test/train/validation>/spect/*npy')
    non_tunable.add_argument('--model_name', type=str, default='model.pt', help='custom model name for saving the model'
                                                                                'after training has completed')
    non_tunable.add_argument('--num_workers', type=int, required=True,
                             help='number of threads to use when loading data')

    # extract
    extract_parser = subparsers.add_parser("extract", add_help=True)
    extract_parser.add_argument('--mel_scale', action='store_true')
    extract_parser.add_argument('--n_fft', required=True, type=int)
    extract_parser.add_argument('--data_dir', required=True, type=str)
    extract_parser.add_argument('--output_data_path', required=True, type=str)
    extract_parser.add_argument('--vert_trim', required=False, default=None)
    extract_parser.add_argument('--random_seed', type=int, default=42,
                                help='what number to use as the random seed. Default:'
                                     ' Deep Thought\'s answer')
    # label
    label_parser = subparsers.add_parser("label", add_help=True)
    label_parser.add_argument('--wav_file', required=True, type=str,
                              help='which .wav file to analyze')
    label_parser.add_argument('--output_csv_path', required=True, type=str,
                              help='where to save the labels')
    # viz
    viz_parser = subparsers.add_parser("viz", add_help=True)
    viz_parser.add_argument('--debug_data_path', type=str,
                            help='location of debugging data',
                            required=True)
    viz_parser.add_argument('--sample_rate', type=int, default=48000,
                            help='sample rate of audio recording')
    viz_parser.add_argument('--hop_length', type=int, default=200,
                            help='length of hops b/t subsequent spectrogram windows')
    viz_parser.add_argument('--apply_heuristics', action='store_true',
                            help='whether or not to apply heuristics specified in heuristics.py'
                                 'this will run the hmm on the processed predictions'
                                 '(this will NOT change results, just visualize them!)')
    return ap


def main():
    args = parser().parse_args()
    if args.command == 'label':
        from simple_labeler import main
        main(args)
    elif args.command == 'train':
        from train import main
        main(args)
    elif args.command == 'extract':
        from extract_data import main
        main(args)
    elif args.command == 'viz':
        from interactive_plot import main
        main(args)
    elif args.command == 'infer':
        from infer import main
        main(args)
