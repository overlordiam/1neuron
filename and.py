## add the data for and gate and call perceptron clas
from utils.model import Perceptron
from utils.all_utils import prepare_data
import pandas as pd

AND = {
    "x1": [0,0,1,1],
    "x2": [0,1,0,1],
    "y": [0,0,0,1]
}

df = pd.DataFrame(AND)

X, y = prepare_data(df)

ETA = 0.3
EPOCHS = 10

model = Perceptron(epochs=EPOCHS, eta=ETA)
model.fit(X, y)

