import pandas as pd
from ebrain_train.Logger import Log


class Validation:
    def __init__(self):
        self.logger = Log()

    def validate(self, filename):
        self.filename = filename
        log_file = open('prediction_logs/01-validation.txt', 'a+')
        log_file.write(self.filename + '\n')
        try:
            data = pd.read_csv(self.filename)
            self.logger.log(log_file, 'Data Validated Successfully')
        except Exception as e:
            self.logger.log(log_file, 'Invalid  Data %s' % e)
        log_file.write('-' * 150 + '\n')
        return
