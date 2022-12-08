from sacred import Experiment

extract_experiment = Experiment()


@extract_experiment.config
def config():

    csv_file = "/Users/wheelerlab/beetles_testing/Trial172_M116_F167-07232021/180102_0381S12.csv"
    wav_file = "/Users/wheelerlab/beetles_testing/Trial172_M116_F167-07232021/180102_0381S12.wav"
    output_data_path = "/tmp/extracted_test/"
