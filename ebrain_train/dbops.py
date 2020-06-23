import pandas as pd
import sqlite3
from ebrain_train.Logger import Log


class Insertion:
    def __init__(self):
        self.logger = Log()

    def insert(self, filename):
        self.filename = filename
        log_file = open('training_logs/03-insertion.txt', 'a+')
        log_file.write(self.filename + '\n')
        try:
            con = sqlite3.connect('database.db')
            self.logger.log(log_file, 'Connected to DB')
        except ConnectionError:
            self.logger.log(log_file, 'Failed to Connect database %s' % ConnectionError)
        try:
            self.data = pd.read_csv(self.filename)
            self.data.to_sql('data', con, if_exists='append', index=False)
            con.close()
            self.logger.log(log_file, 'Data stored in DB')
        except Exception as e:
            self.logger.log(log_file, 'Failed to Insert Data %s' % e)
        log_file.write('-' * 150 + '\n')
        return
