import wget
import shutil
import pandas as pd
import os
import sys
import tarfile
from sklearn import datasets


def create_imdb_csv(out_dir):
    def bar_progress(current, total, width=80):
        progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100,
                                                                  current, total)
        sys.stdout.write("\r" + progress_message)
        sys.stdout.flush()

    filename = wget.download('https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
                             'data', bar=bar_progress)
    print('\nUnzipping')
    file = tarfile.open(filename)
    file.extractall(out_dir)
    file.close()
    os.remove(filename)
    classes = None
    data = {}
    for mode in ['train', 'test']:
        folder = os.path.join(out_dir, 'aclImdb/') + mode
        print('Removing files for ' + mode)
        for file in os.listdir(folder):
            filepath = os.path.join(folder, file)
            if os.path.isdir(filepath) and file == 'unsup':
                shutil.rmtree(filepath)
            elif not os.path.isdir(filepath):
                os.remove(filepath)
        print('Creating csv for ' + mode)
        d = datasets.load_files(os.path.join(out_dir, 'aclImdb/') + mode, categories=classes, encoding='utf8')
        data[mode] = pd.concat([pd.Series(d.data), pd.Series(d.target)], axis=1, ignore_index=True)
        classes = d.target_names
        data[mode].columns = ['text', 'label']
        data[mode].to_csv(os.path.join(out_dir, 'IMDB_') + mode + '.csv', index=False)
    shutil.rmtree(os.path.join(out_dir, 'aclImdb/'))
