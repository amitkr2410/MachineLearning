from enum import Enum
from dataclasses import dataclass
from pathlib import Path

class Criteria(Enum):
    LOSS = "loss"
    ACCURACY ="accuracy"
    F1 = "f1"

@dataclass
class AllParameters:
    parent_dir: Path = Path('')
    #Parameters
    model_name: str="vgg16_pretrained_true" #[vgg16_pretrained_true, vgg16_pretrained_false,
                                            #  ,vgg16_custom, cnn_4layers_custom]
    num_classes: int = 10                     #2 (brain-tumor), 7(FaceSmile), 102(Flowers)
    device_name = 'cpu' # 'cuda' or 'cpu'
    #PreprocessParameters
    seed: int =1
    test_size: float = 0.2
    train_folder: Path = Path('data/train')
    test_folder: Path  = Path('data/test')
    predict_folder: Path = Path('data/predict')
    
    #TrainParameters
    learning_rate: float = 0.002
    weight_decay: float = 0.2
    batch_size: int = 10
    num_epochs: int = 20
    
    #EvaluationMetrics
    #criteria: Criteria
    loss: float = 0.0
    accuracy: float = 0.0
    f1: float = 0.0
    



