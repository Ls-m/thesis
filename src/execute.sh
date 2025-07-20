

cd src tensorboard --logdir logs/

python train.py --config ../configs/improved_config.yaml --override model.name=RWKV --override training.max_epochs=150 --override data.csv_folder=bidmc --override  data.sampling_rate=125 --override data.segment_length=1000 --override preprocessing.downsample.target_rate=25 --override model.input_size=200 --fold subject_50 --override logging.experiment_name=RWKV_b64_e150_bidmc_experiment --override model.dropout=0.15


python train.py --config ../configs/improved_config.yaml --override model.name=ImprovedTransformer --override training.max_epochs=150 --override data.csv_folder=bidmc --override  data.sampling_rate=125 --override data.segment_length=1000 --override preprocessing.downsample.target_rate=25 --override model.input_size=200 --fold subject_50 --override logging.experiment_name=ImprovedTransformer_b64_e150_bidmc_experiment --override model.dropout=0.15


python train.py --config ../configs/improved_config.yaml --override model.name=WaveNet --override training.max_epochs=400 --override data.csv_folder=bidmc --override  data.sampling_rate=125 --override data.segment_length=1000 --override preprocessing.downsample.target_rate=25 --override model.input_size=200 --fold subject_50 --override logging.experiment_name=WaveNet_b64_e400_bidmc_experiment --override model.dropout=0.15 --override training.patience=80




python train_subject_wise.py --config ../configs/improved_config.yaml --override model.name=ImprovedTransformer --override training.max_epochs=150 --override data.csv_folder=bidmc --override  data.sampling_rate=125 --override data.segment_length=1000 --override preprocessing.downsample.target_rate=25 --override model.input_size=200 --fold subject_50 --override logging.experiment_name=ImprovedTransformer_b64_e150_bidmc_vsep_experiment --override model.dropout=0.2

python train_subject_wise.py --config ../configs/improved_config.yaml --override model.name=RWKV --override training.max_epochs=150 --override data.csv_folder=bidmc --override  data.sampling_rate=125 --override data.segment_length=1000 --override preprocessing.downsample.target_rate=25 --override model.input_size=200 --fold subject_50 --override logging.experiment_name=RWKV_b64_e150_bidmc_vsep_experiment --override model.dropout=0.2

python train_subject_wise.py --config ../configs/improved_config.yaml --override model.name=WaveNet --override training.max_epochs=150 --override data.csv_folder=bidmc --override  data.sampling_rate=125 --override data.segment_length=1000 --override preprocessing.downsample.target_rate=25 --override model.input_size=200 --fold subject_50 --override logging.experiment_name=WaveNet_b64_e150_bidmc_vsep_experiment --override model.dropout=0.2 --override training.patience=50

python train_subject_wise.py --config ../configs/improved_config.yaml --override model.name=RWKV --override training.max_epochs=150 --override data.csv_folder=bidmc --override  data.sampling_rate=125 --override data.segment_length=1000 --override preprocessing.downsample.target_rate=25 --override model.input_size=200 --fold subject_50 --override logging.experiment_name=RWKV_b64_e150_bidmc_experiment --override model.dropout=0.2 --override training.patience=70



python train_subject_wise.py --config ../configs/improved_config.yaml --override model.name=ImprovedTransformer --override training.max_epochs=150 --override data.csv_folder=capno --override  data.sampling_rate=300 --override data.segment_length=2400 --override preprocessing.downsample.target_rate=30 --override model.input_size=240 --fold 0009_8min --override logging.experiment_name=ImprovedTransformer_b64_e150_capno_experiment --override model.dropout=0.2
