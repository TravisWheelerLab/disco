from beetles.train import train_func

from test_tube.hpc import SlurmCluster, HyperOptArgumentParser


def train(args, kwargs):
    train_func(args)


def setup_hyperparser():
    parser = HyperOptArgumentParser(strategy='random_search')

    parser.opt_range('--learning_rate',
                     type=float,
                     default=1e-3,
                     tunable=True,
                     low=1e-4,
                     high=5e-2,
                     log_base=10,
                     nb_samples=30,
                     )
    parser.opt_list('--n_fft',
                    type=int,
                    default=650,
                    tunable=True,
                    options=[650, 750, 850, 950, 1050, 1150, 1250]
                    )
    parser.opt_range('--vertical_trim',
                     type=int,
                     default=0,
                     tunable=False,
                     low=0,
                     high=40,
                     nb_samples=5,
                     )
    parser.opt_range('--begin_mask',
                     type=int,
                     default=0,
                     tunable=False,
                     low=20,
                     high=30,
                     nb_samples=2,
                     )
    parser.opt_range('--end_mask',
                     type=int,
                     default=10,
                     tunable=False,
                     low=10,
                     high=20,
                     nb_samples=5,
                     )

    # I still really dislike this API because it requires copying and pasting argument parsers
    parser.add_argument('--test_tube_exp_name', default='my_test')
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--mel', action='store_true')
    parser.add_argument('--apply_attn', action='store_true')
    parser.add_argument('--bootstrap', action='store_true')
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--tune_initial_lr', action='store_true')
    parser.add_argument('--gpus', type=int, required=True)
    parser.add_argument('--num_nodes', type=int, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--check_val_every_n_epoch', type=int, required=False,
                        default=1)
    parser.add_argument('--log_dir', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='model.pt')
    parser.add_argument('--num_workers', type=int, required=True)
    parser.add_argument('--mask_beginning_and_end', action='store_true',
                        help='whether or not to mask the beginning and end of sequences')
    return parser.parse_args()


if __name__ == '__main__':
    hyperparams = setup_hyperparser()

    # init cluster
    cluster = SlurmCluster(
        hyperparam_optimizer=hyperparams,
        log_path=hyperparams.log_dir,
        python_cmd='python',
    )

    cluster.add_command("source /home/tc229954/anaconda/bin/activate")
    cluster.add_command("conda activate beetles")

    cluster.add_slurm_cmd(cmd='partition',
                          value='wheeler_lab_gpu',
                          comment='partition')

    cluster.per_experiment_nb_gpus = hyperparams.gpus

    cluster.optimize_parallel_cluster_gpu(train,
                                          nb_trials=30,
                                          job_name='hparam_tuner',
                                          job_display_name='tune',
                                          )
