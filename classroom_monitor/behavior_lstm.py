import os
import torch
import torch.nn as nn
import threading
import numpy as np
from typing import Tuple, Optional

class BehaviorLSTM(nn.Module):
    """
    LSTM Model for Classifying Student Behaviors from a sequence of Pose Keypoints.
    
    Input shape: (batch_size, seq_len, input_dim)
    input_dim = num_keypoints * 2 (x,y) + auxiliary_features
    """
    def __init__(self, input_dim=34, hidden_dim=64, num_layers=2, num_classes=5):
        super(BehaviorLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        # Bidirectional LSTM to capture motion context better
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=0.2 if num_layers > 1 else 0.0,
            bidirectional=True
        )
        
        # Output layers
        self.fc1 = nn.Linear(hidden_dim * 2, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        
        # lstm_out shape: (batch, seq_len, hidden_dim * 2)
        lstm_out, (hn, cn) = self.lstm(x)
        
        # We take the output from the final time step
        # Since it's bidirectional, we concatenate the forward and backward hidden states from the last layer
        last_hidden = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1)
        
        out = self.fc1(last_hidden)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class SharedBehaviorLSTM:
    """
    Process-wide singleton holding the loaded LSTM model.
    Similar to _SharedCNNModel for fight detection.
    """
    _lock = threading.Lock()
    _model = None
    _device = None
    _model_loaded = False
    
    # Class mapping
    # 0: focused, 1: distracted, 2: hand_raised, 3: using_phone, 4: eating_food
    CLASSES = ['focused', 'distracted', 'hand_raised', 'using_phone', 'eating_food']

    @classmethod
    def get(cls, input_dim=38): # 17 keypoints * 2 (x,y) + 4 aux features (phone_dist, food_dist, book_dist, writing_var)
        with cls._lock:
            if cls._model is None:
                cls._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                cls._model = BehaviorLSTM(input_dim=input_dim, hidden_dim=64, num_layers=2, num_classes=len(cls.CLASSES))
                
                weights_path = os.environ.get('BEHAVIOR_LSTM_WEIGHTS_PATH')
                if not weights_path:
                    weights_dir = os.path.join(os.path.dirname(__file__), 'model_weights')
                    weights_path = os.path.join(weights_dir, 'behavior_lstm_weights.pth')
                
                if os.path.exists(weights_path):
                    try:
                        state_dict = torch.load(weights_path, map_location=cls._device, weights_only=True)
                        cls._model.load_state_dict(state_dict)
                        cls._model_loaded = True
                        print(f'[BEHAVIOR-LSTM] Fine-tuned weights loaded from {weights_path}')
                    except Exception as e:
                        print(f'[WARN] Failed to load LSTM weights: {e}')
                else:
                    print(f'[WARN] LSTM weights not found at {weights_path}. Model will run in fallback mode.')
                
                cls._model.to(cls._device)
                cls._model.eval()
            
            return cls._model, cls._device, cls._model_loaded

