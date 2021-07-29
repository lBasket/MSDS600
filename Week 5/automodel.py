from argparse import ArgumentParser
import pandas as pd
from pycaret.classification import setup, compare_models, predict_model, save_model, load_model, plot_model
import pickle
from churn_cleaner import InitAttributeCleaner


def get_args():
    parser = ArgumentParser()

    parser.add_argument(
            'filename',
            help='Name of file of feature data to predict.'
        )
    parser.add_argument(
            '-clean',
            help='Whether or not to clean the data. Default True.',
            default=True,
            required=False
        )
    
    return parser

def process_args(args):
    process_status = True

    try:
        df = pd.read_csv('./' + args.filename)
    except FileNotFoundError:
        process_status = False
        err_mesg = 'File not found.'

    return args, df, process_status


class auto_predictor():
    def __init__(self, df):
        self.df = df
        self.model = load_model('best_model')

    def clean(self):
        cleaner = InitAttributeCleaner()
        self.df = cleaner.fit_transform(self.df)
    
    def predict(self):
        self.preds = self.model.predict(self.df)

    def print_preds(self):
        for i in range(0, len(self.preds)):
            print(f'{i} | Predicted class {self.preds[i]}')

    def get_preds(self):
        return self.preds


if __name__ == '__main__':
    arg_parser = get_args()
    args, df, arg_status = process_args(arg_parser.parse_args())

    if arg_status:
        ap = auto_predictor(df)
        
        if args.clean:
           ap.clean()
        
        ap.predict()

        ap.print_preds()

    else:
        print('File parsing failed or bad arguments passed.')
