## add the data for or gate and call perceptron class
from utils.model import Perceptron
from utils.all_utils import prepare_data, save_model, save_plot
import pandas as pd
import logging
import os

logging_dir = "logs"
os.makedirs(logging_dir, exist_ok=True)
logging_str = "[%(asctime)s: %(levelname)s %(module)s] %(message)s"
logging.basicConfig(filename=os.path.join(logging_dir, "running_logs_or.log"), 
                    level=logging.INFO, format=logging_str, filemode='a')

def main(df, eta, epochs, filename, plotFileName):
    df = pd.DataFrame(OR)
    logging.info(f"The dataframe is: \n{df}")

    X, y = prepare_data(df)

    

    model = Perceptron(epochs=EPOCHS, eta=ETA)
    model.fit(X, y)

    _ = model.total_loss()

    save_model(model, filename=filename)
    save_plot(df, file_name=plotFileName, model=model)

if __name__ == "__main__":
    OR = {
    "x1": [0,0,1,1],
    "x2": [0,1,0,1],
    "y": [0,1,1,1]
    }

    ETA = 0.3
    EPOCHS = 100

    try:
        logging.info("<<<<<<<< STARTED TRAINING >>>>>>>>")
        main(df=OR, eta=ETA, epochs=EPOCHS, filename="or.model", plotFileName="or.png")
        logging.info(">>>>>>> ENDED TRAINING <<<<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e