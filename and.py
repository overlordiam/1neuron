## add the data for and gate and call perceptron clas
from utils.model import Perceptron
from utils.all_utils import prepare_data, save_model, save_plot
import pandas as pd

def main(df, eta, epochs, filename, plotFileName):
    
    df = pd.DataFrame(df)
    print(df)

    X, y = prepare_data(df)

    model = Perceptron(epochs=epochs, eta=eta)
    model.fit(X, y)

    _ = model.total_loss()

    save_model(model, filename=filename)
    save_plot(df, file_name=plotFileName, model=model)

if __name__ == "__main__":

    AND = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y": [0,0,0,1]
    }

    ETA = 0.3
    EPOCHS = 10

    main(df=AND, eta=ETA, epochs=EPOCHS, filename="and.model", plotFileName="and.png")