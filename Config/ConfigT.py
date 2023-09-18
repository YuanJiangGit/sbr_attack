from configparser import ConfigParser


class MyConf:
    def __init__(self, config_file):
        config = ConfigParser()
        config.read(config_file, encoding='utf-8')
        self._config = config
        self.config_file=config_file

        # config.write(open(config_file,'w'))

    @property
    def WordEmbeddingPath(self):
        return self._config.get('data', 'WordEmbeddingPath')

    @property
    def log_to_txt(self):
        return self._config.get('data', 'log_to_txt')

    @property
    def log_to_csv(self):
        return self._config.get('data', 'log_to_csv')

    @property
    def enable_visdom(self):
        return self._config.getboolean('data', 'enable_visdom')

    @property
    def enable_wandb(self):
        return self._config.getboolean('data', 'enable_wandb')

    @property
    def disable_stdout(self):
        return self._config.getboolean('data', 'disable_stdout')

    @property
    def num_examples(self):
        return self._config.getint('data', 'num_examples')

    @property
    def csv_style(self):
        return self._config.get('data', 'csv_style')


if __name__ == '__main__':
    conf = MyConf('./config.cfg')
    print(conf.num_examples)
