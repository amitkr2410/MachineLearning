from enum import Enum
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AllParameters:
    parent_dir: Path = Path('')
    #Parameters
    model_name: str="cnn_4layers_custom" #[vgg16_pretrained_true, vgg16_pretrained_false,
                                            #  ,vgg16_custom, cnn_4layers_custom
                                            #, cnn_with_attention, only_attention, ]
    num_classes: int = 2                     #2 (brain-tumor), 7(FaceSmile), 102(Flowers)
    #device_name = 'cpu' # 'cuda' or 'cpu'
    #PreprocessParameters
    seed: int =1
    test_size: float = 0.2
    train_folder: Path = Path('data/train')
    test_folder: Path  = Path('data/test')
    predict_folder: Path = Path('data/predict')
    
    #TrainParameters
    learning_rate: float = 0.002
    weight_decay: float = 0.2
    batch_size: int = 100
    num_epochs: int = 2
    
    #Save model after training
    save_model_flag: str='yes'
    save_model_dir: str='final_model'
    save_model_filename: str='cnn_4layers_custom.pth'
    save_parameters_file: str='cnn_4layers_results'
    #EvaluationMetrics
    #criteria: Criteria
    train_accuracy: float = 0.0
    test_val_loss: float = 0.0
    test_accuracy: float = 0.0

    



