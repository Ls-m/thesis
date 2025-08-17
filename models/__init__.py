from .cnn1d import CNN1D, ResidualCNN1D
from .lstm import LSTM, BiLSTM
from .cnn_lstm import CNN_LSTM, AttentionCNN_LSTM
from .rwkv import RWKV, ImprovedTransformer, WaveNet,RWKV_DualBranch


def get_model(model_name: str, **kwargs):
    """Factory function to get model by name."""
    
    models = {
        'CNN1D': CNN1D,
        'ResidualCNN1D': ResidualCNN1D,
        'LSTM': LSTM,
        'BiLSTM': BiLSTM,
        'CNN_LSTM': CNN_LSTM,
        'AttentionCNN_LSTM': AttentionCNN_LSTM,
        'RWKV': RWKV,
        'ImprovedTransformer': ImprovedTransformer,
        'WaveNet': WaveNet,
        'RWKV_DualBranch': RWKV_DualBranch
    }
    
    if model_name not in models:
        raise ValueError(f"Model {model_name} not found. Available models: {list(models.keys())}")
    
    return models[model_name](**kwargs)


__all__ = [
    'CNN1D', 'ResidualCNN1D', 'LSTM', 'BiLSTM', 
    'CNN_LSTM', 'AttentionCNN_LSTM', 'RWKV', 
    'ImprovedTransformer', 'WaveNet', 'RWKV_DualBranch', 'get_model'
]
