from sacred import Experiment

label_experiment = Experiment()


@label_experiment.config
def config():
    wav_file = "/Users/wheelerlab/beetles_testing/unannotated_files_12_12_2022/Trial155_M113_F115-07202021/000102_0280S34D06.wav"
    output_csv_path = "/tmp/fork"
