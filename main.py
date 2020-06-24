from ebrain_train.validation import Validator
from ebrain_train.cleaner import Filter
from ebrain_train.dbops import Insertion
from ebrain_train.dataprocess import Preprocess
from ebrain_train.model import Model_Selector
from sklearn.model_selection import train_test_split
from ebrain_train.Logger import Log
from ebrain_predict.prediction_validation import Validation
from ebrain_predict.prediction_cleaner import Prediction_Filter
import pickle


class Train:
    def __init__(self):
        self.validator = Validator()
        self.filter = Filter()
        self.insertion = Insertion()
        self.process = Preprocess()
        self.model = Model_Selector()

    def best_model(self, raw_file):
        self.validator.validate(raw_file)
        self.filter.transform(raw_file)

        self.insertion.insert('transformed_data/' + raw_file)
        df = self.process.export_data('database.db')
        data = self.process.cost(df)
        data = self.process.rating(data)
        data = self.process.table(data)
        X, y = self.process.split(data)
        X = self.process.encoder(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        model = self.model.best_model(X_train, X_test, y_train, y_test)
        return model


class Prediction:
    def __init__(self):
        self.logger = Log()
        self.a = Validation()
        self.b = Prediction_Filter()
        self.c = Preprocess()
        with open('models/model-0.8266616494366493.pkl', 'rb') as file:
            self.model = pickle.load(file)

    def predict(self, filename):
        self.a.validate(filename)
        self.data = self.b.transform(filename)
        self.data = self.c.rating(self.data)
        self.data = self.c.table(self.data)
        self.X, self.y = self.c.split(self.data)
        self.X = self.c.encoder(self.X)
        prediction = self.model.predict(self.X)
        return prediction
