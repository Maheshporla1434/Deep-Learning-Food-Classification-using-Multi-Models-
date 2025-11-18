import logging
import sys
class Logger:
    def get_logs(log_name):
        try:
            logger = logging.getLogger(log_name)
            logger.setLevel(logging.DEBUG)

            handler = logging.FileHandler(f'C:\\Users\\Mahesh Porla\\Downloads\\foodproject\log_files\\{log_name}.log')
            formate = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

            handler.setFormatter(formate)
            logger.addHandler(handler)

            return logger
        except Exception as e:
            exc_type, exc_msg, exc_line = sys.exc_info()

            print(f'{exc_type} at {exc_line.tb_lineno} as {exc_msg}')