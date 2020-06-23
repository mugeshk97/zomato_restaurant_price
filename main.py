from ebrain_train.validation import Validator
from ebrain_train.cleaner import Filter
from ebrain_train.dbops import Insertion
from ebrain_train.dataprocess import Preprocess
from ebrain_train.model import Model_Selector
from sklearn.model_selection import train_test_split


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
