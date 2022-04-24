import logging
import yaml


def load_logger(log_dir, log_level):
    logger = logging.getLogger('CSKG')
    if log_level == 'INFO':
        lv = logging.INFO
    elif log_level == 'ERROR':
        lv = logging.ERROR
    elif log_level == 'DEBUG':
        lv = logging.DEBUG
    else:
        raise NotImplementedError
    logger.setLevel(lv)

    formatter = logging.Formatter('%(asctime)s [%(name)s] [%(levelname)s] :: %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_dir)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger

def load_yaml(f):
    if type(f) is str:
        with open(f, 'r') as fp:
            config = yaml.load(fp, Loader=yaml.FullLoader)
    else:
        raise NotImplementedError

    return config