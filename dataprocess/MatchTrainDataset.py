import os
import pandas as pd

datasets = {
    # 'ambari': 'Ambari2.csv',
    # 'camel': 'Camel2.csv',
    'chromium': 'Chromium2.csv',
    # 'derby': 'Derby2.csv',
    # 'wicket': 'Wicket2.csv',
}

LTRWES_training = {
    'ambari': 'ambari.csv',
    'camel': 'camel.csv',
    'chromium': 'chromium.csv',
    'derby': 'derby.csv',
    'wicket': 'wicket.csv',
}

FARSEC_training = {
    'ambari': 'ambari-clnisq.csv',
    'camel': 'camel-two.csv',
    'chromium': 'chromium-clni.csv',
    'derby': 'derby-clnisq.csv',
    'wicket': 'wicket-clnitwo.csv',
}

for project in datasets.keys():
    data_file = os.path.join('..', 'resources', 'dataset', datasets[project])
    data_all = pd.read_csv(data_file, sep=',', encoding='ISO-8859-1')
    # LTRWES_train_file = os.path.join('..', 'resources', 'LTRWES_training', LTRWES_training[project])
    # LTRWES_train_all = pd.read_csv(LTRWES_train_file, sep=',', encoding='ISO-8859-1')
    # train_match = data_all[data_all.issue_id.map(lambda x: x in LTRWES_train_all.issue_id.tolist())]
    # LTRWES_train_save = os.path.join('..', 'resources', 'LTRWES_training', 'origin', datasets[project])
    # if not os.path.exists(LTRWES_train_save):
    #     train_match.to_csv(LTRWES_train_save, encoding='utf-8')

    FARSEC_train_file = os.path.join('..', 'resources', 'FARSEC_training', FARSEC_training[project])
    FARSEC_train_all = pd.read_csv(FARSEC_train_file, sep=',', encoding='ISO-8859-1')
    train_match = data_all[data_all.id.map(lambda x: x in FARSEC_train_all.id.tolist())]
    FARSEC_train_save = os.path.join('..', 'resources', 'FARSEC_training', 'origin', datasets[project])
    if not os.path.exists(FARSEC_train_save):
        train_match.to_csv(FARSEC_train_save, encoding='utf-8')
