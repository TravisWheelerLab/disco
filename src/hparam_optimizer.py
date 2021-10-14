from train import train_func

from test_tube.hpc import SlurmCluster, HyperOptArgumentParser


def train(args, kwargs):
    train_func(args)


def setup_hyperparser():
    parser = HyperOptArgumentParser(strategy='random_search')

    parser.opt_range('--learning_rate',
                     type=float,
                     default=1e-3,
                     tunable=True,
                     low=1e-7,
                     high=1e-1,
                     log_base=10,
                     nb_samples=5,
                     )

    parser.add_argument('--test_tube_exp_name', default='my_test')
    parser.add_argument('--log_path', default='/some/path/to/log')
    parser.add_argument('--vert_trim', required=True, type=int)
    parser.add_argument('--n_fft', required=True, type=int)
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--mel', action='store_true')
    parser.add_argument('--bootstrap', action='store_true')
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--tune_initial_lr', action='store_true')
    parser.add_argument('--in_channels', type=int, required=True)
    parser.add_argument('--gpus', type=int, required=True)
    parser.add_argument('--num_nodes', type=int, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--check_val_every_n_epoch', type=int, required=False,
                        default=1)
    parser.add_argument('--log_dir', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='model.pt')
    parser.add_argument('--num_workers', type=int, required=True)

    return parser.parse_args()


if __name__ == '__main__':
    hyperparams = setup_hyperparser()

    # init cluster
    cluster = SlurmCluster(
        hyperparam_optimizer=hyperparams,
        log_path='/home/tc229954/beetles-logs/',
        python_cmd='python',
    )

    cluster.add_command("source /home/tc229954/anaconda/bin/activate")
    cluster.add_command("conda activate beetles")

    cluster.add_slurm_cmd(cmd='partition', value='wheeler_lab_gpu', comment='partition')

    cluster.per_experiment_nb_gpus = 1
    cluster.per_experiment_nb_nodes = 1

    cluster.optimize_parallel_cluster_gpu(train,
                                          nb_trials=20,
                                          job_name='hparam_tuner',
                                          job_display_name='tune',
                                          )
