## add the data for or gate and call perceptron class
from utils.model import Perceptron
from utils.all_utils import prepare_data, save_model, save_plot
import pandas as pd

def main(df, eta, epochs, filename, plotFileName):
    df = pd.DataFrame(OR)
    print(df)

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
    EPOCHS = 10

    main(df=OR, eta=ETA, epochs=EPOCHS, filename="or.model", plotFileName="or.png")
