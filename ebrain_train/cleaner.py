import pandas as pd
from ebrain_train.Logger import Log


class Filter:
    def __init__(self):
        self.logger = Log()

    def transform(self, filename):
        self.filename = filename
        self.dataframe = pd.read_csv(self.filename)
        log_file = open('training_logs/02-transformer.txt', 'a+')
        log_file.write(self.filename + '\n')
        try:
            self.dataframe = self.dataframe.rename(
                columns={'approx_cost(for two people)': 'cost', 'listed_in(type)': 'type',
                         'listed_in(city)': 'city', 'menu_item': 'menu', 'reviews_list': 'review'})
            self.logger.log(log_file, 'Column Names are changed')
        except Exception as e:
            self.logger.log(log_file, 'Failed to Rename the columns %s' % e)
        try:
            self.dataframe.drop(['address', 'url', 'phone', 'dish_liked', 'review', 'city'], axis=1, inplace=True)
            self.logger.log(log_file, 'Removed Unnecessary Features')
        except Exception as e:
            self.logger.log(log_file, 'Failed to remove unnecessary features %s' % e)
        try:
            self.dataframe.drop_duplicates(inplace=True)
            self.logger.log(log_file, 'Removed Duplicate Values')
        except Exception as e:
            self.logger.log(log_file, 'Failed to drop duplicates %s' % e)
        try:
            self.dataframe.dropna(how='any', inplace=True)
            self.logger.log(log_file, 'Removed Null Values')
        except Exception as e:
            self.logger.log(log_file, 'Failed to Drop Null values %s' % e)
        log_file.write('-' * 150 + '\n')
        return self.dataframe.to_csv('transformed_data/' + self.filename, index=False)
