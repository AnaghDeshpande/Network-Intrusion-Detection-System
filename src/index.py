# from .model import *
# from .train_pipline import *
# from .data_processor import *

from src.model import *
from src.train_pipeline import *
from src.data_processor import *

def run_model():
    train_path = "data/NSL-KDD/KDDTrain+.txt"
    model, preproc, label_enc, history, results = train_pipeline(train_path, epochs=50, 
                                            batch_size=128, seq_len=10, model_save_path="nids_cnn_bilstm.h5")
    print("Results:", results)

    return model, preproc, label_enc, history, results

if __name__ == "__main__":
    run_model()
    