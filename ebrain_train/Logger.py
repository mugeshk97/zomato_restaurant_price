import datetime


class Log:
    def __init__(self):
        pass

    def log(self, file, message):
        self.now = datetime.datetime.now()
        self.date = self.now.date()
        self.current_time = self.now.strftime("%H:%M:%S")
        file.write(str(self.date) + "/" + str(self.current_time) + "\t\t" + message + '\n')
