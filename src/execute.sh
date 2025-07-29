

cd src tensorboard --logdir logs/

python train.py --config ../configs/improved_config.yaml --override model.name=RWKV --override training.max_epochs=150 --override data.csv_folder=bidmc --override  data.sampling_rate=125 --override data.segment_length=1000 --override preprocessing.downsample.target_rate=25 --override model.input_size=200 --fold subject_50 --override logging.experiment_name=RWKV_b64_e150_bidmc_experiment --override model.dropout=0.15


python train.py --config ../configs/improved_config.yaml --override model.name=ImprovedTransformer --override training.max_epochs=150 --override data.csv_folder=bidmc --override  data.sampling_rate=125 --override data.segment_length=1000 --override preprocessing.downsample.target_rate=25 --override model.input_size=200 --fold subject_50 --override logging.experiment_name=ImprovedTransformer_b64_e150_bidmc_experiment --override model.dropout=0.15


python train.py --config ../configs/improved_config.yaml --override model.name=WaveNet --override training.max_epochs=400 --override data.csv_folder=bidmc --override  data.sampling_rate=125 --override data.segment_length=1000 --override preprocessing.downsample.target_rate=25 --override model.input_size=200 --fold subject_50 --override logging.experiment_name=WaveNet_b64_e400_bidmc_experiment --override model.dropout=0.15 --override training.patience=80




python train_subject_wise.py --config ../configs/improved_config.yaml --override model.name=ImprovedTransformer --override training.max_epochs=150 --override data.csv_folder=bidmc --override  data.sampling_rate=125 --override data.segment_length=1000 --override preprocessing.downsample.target_rate=25 --override model.input_size=200 --fold subject_50 --override logging.experiment_name=ImprovedTransformer_b64_e150_bidmc_vsep_experiment --override model.dropout=0.2

python train_subject_wise.py --config ../configs/improved_config.yaml --override model.name=RWKV --override training.max_epochs=150 --override data.csv_folder=bidmc --override  data.sampling_rate=125 --override data.segment_length=1000 --override preprocessing.downsample.target_rate=25 --override model.input_size=200 --fold subject_50 --override logging.experiment_name=RWKV_b64_e150_bidmc_vsep_experiment --override model.dropout=0.2

python train_subject_wise.py --config ../configs/improved_config.yaml --override model.name=WaveNet --override training.max_epochs=150 --override data.csv_folder=bidmc --override  data.sampling_rate=125 --override data.segment_length=1000 --override preprocessing.downsample.target_rate=25 --override model.input_size=200 --fold subject_50 --override logging.experiment_name=WaveNet_b64_e150_bidmc_vsep_experiment --override model.dropout=0.2 --override training.patience=50

python train_subject_wise.py --config ../configs/improved_config.yaml --override model.name=RWKV --override training.max_epochs=150 --override data.csv_folder=bidmc --override  data.sampling_rate=125 --override data.segment_length=1000 --override preprocessing.downsample.target_rate=25 --override model.input_size=200 --fold subject_50 --override logging.experiment_name=RWKV_b64_e150_bidmc_experiment --override model.dropout=0.2 --override training.patience=70



python train_enhanced.py --config ../configs/improved_config.yaml --override model.name=AttentionCNN_LSTM --override training.max_epochs=50 --override data.csv_folder=capno --override  data.sampling_rate=300 --override data.segment_length=2400 --override preprocessing.downsample.target_rate=30 --override model.input_size=240 --override logging.experiment_name=AttentionCNN_LSTM_debugged_b64_d4_e50_capno_5f_experiment --override model.dropout=0.4 --cv-method k_fold --n-folds 5


python train_enhanced.py --config ../configs/improved_config.yaml --override model.name=ImprovedTransformer --override training.max_epochs=50 --override data.csv_folder=capno --override  data.sampling_rate=300 --override data.segment_length=2400 --override preprocessing.downsample.target_rate=30 --override model.input_size=240 --override logging.experiment_name=ImprovedTransformer_debugged_b64_d5_capno_5f_experiment --override model.dropout=0.5 --override preprocessing.bandpass_filter.high_freq=1.0 --cv-method k_fold --n-folds 5


python train_enhanced.py --config ../configs/improved_config.yaml --override model.name=ImprovedTransformer --override training.max_epochs=70 --override data.csv_folder=capno --override  data.sampling_rate=300 --override data.segment_length=2400 --override preprocessing.downsample.target_rate=30 --override model.input_size=240 --override logging.experiment_name=ImprovedTransformer_d6_capno_experiment_v1 --override model.dropout=0.6 --override preprocessing.bandpass_filter.high_freq=0.6 --cv-method k_fold --n-folds 5



python train_enhanced.py --config ../configs/improved_config.yaml --override model.name=ImprovedTransformer --override model.num_layers=4 --override training.max_epochs=70 --override data.csv_folder=capno --override  data.sampling_rate=300 --override data.segment_length=2400 --override preprocessing.downsample.target_rate=30 --override model.input_size=240 --override logging.experiment_name=ImprovedTransformer_d6_capno_experiment_v2 --override model.dropout=0.6 --override preprocessing.bandpass_filter.high_freq=0.6 --cv-method k_fold --n-folds 5


python train_enhanced.py --config ../configs/improved_config.yaml --override model.name=ImprovedTransformer --override model.num_layers=4 --override training.max_epochs=70 --override data.csv_folder=capno --override  data.sampling_rate=300 --override data.segment_length=2400 --override preprocessing.downsample.target_rate=30 --override model.input_size=240 --override logging.experiment_name=ImprovedTransformer_d6_capno_experiment_v3 --override model.dropout=0.6 --override preprocessing.bandpass_filter.high_freq=0.6 --override preprocessing.bandpass_filter.low_freq=0.1 --cv-method k_fold --n-folds 5

python train_enhanced.py --config ../configs/improved_config.yaml --override model.name=ImprovedTransformer --override model.num_layers=4 --override model.hidden_size=128 --override training.max_epochs=70 --override data.csv_folder=capno --override  data.sampling_rate=300 --override data.segment_length=2400 --override preprocessing.downsample.target_rate=30 --override model.input_size=240 --override logging.experiment_name=ImprovedTransformer_d6_capno_experiment_v4 --override model.dropout=0.6 --override preprocessing.bandpass_filter.high_freq=0.6 --override preprocessing.bandpass_filter.low_freq=0.1 --cv-method k_fold --n-folds 5


python train_enhanced.py --config ../configs/improved_config.yaml --override model.name=ImprovedTransformer --override model.num_layers=4 --override training.max_epochs=70 --override data.csv_folder=capno --override  data.sampling_rate=300 --override data.segment_length=2400 --override preprocessing.downsample.target_rate=30 --override model.input_size=240 --override logging.experiment_name=ImprovedTransformer_d6_capno_experiment_RR_v3 --override model.dropout=0.6 --override preprocessing.bandpass_filter.high_freq=0.6 --override preprocessing.bandpass_filter.low_freq=0.1 --cv-method k_fold --n-folds 5



python train_enhanced.py --config ../configs/improved_config.yaml --override model.name=ImprovedTransformer --override model.num_layers=4 --override training.max_epochs=70 --override data.csv_folder=capno --override  data.sampling_rate=300 --override data.segment_length=2400 --override preprocessing.downsample.target_rate=30 --override model.input_size=240 --override logging.experiment_name=ImprovedTransformer_d6_capno_experiment_v3 --override model.dropout=0.6 --override preprocessing.bandpass_filter.high_freq=0.6 --override preprocessing.bandpass_filter.low_freq=0.1 --cv-method k_fold --n-folds 5


python train_enhanced.py --config ../configs/improved_config.yaml --override model.name=RWKV --override model.num_layers=6 --override training.max_epochs=70 --override data.csv_folder=capno --override  data.sampling_rate=300 --override data.segment_length=2400 --override preprocessing.downsample.target_rate=30 --override model.input_size=240 --override logging.experiment_name=RWKV_update_d6_capno_experiment_v3 --override model.dropout=0.6 --override preprocessing.bandpass_filter.high_freq=0.6 --override preprocessing.bandpass_filter.low_freq=0.1 --cv-method k_fold --n-folds 5