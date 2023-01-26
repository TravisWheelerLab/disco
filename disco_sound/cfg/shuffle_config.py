from disco_sound.cfg import shuffle_experiment


@shuffle_experiment.config
def config():
    data_directory = "/tmp/extracted_test/"
    move = False
