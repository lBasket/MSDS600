from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class InitAttributeCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
            return self # Nothing to do here, apparently
    def transform(self, X, y=None):        
        
        ### customerID : DROP
        X = X.drop(['customerID'], axis=1).copy()

        
        ### New column : average charge
        # Fill TotalCharges with the mode of the dataset because it's so heavily skewed
        X['TotalCharges'] = X['TotalCharges'].fillna(X['TotalCharges'].mode().iloc[0]).copy()
        
        X['average_charge'] = X['TotalCharges'] / X['tenure']
        
        # We've got some infs here so we're going to replace our nulls with our average average charge
        X.loc[X['average_charge'] == np.inf, 'average_charge'] = 0#X['average_charge'].mean()
        
        # Now normalize
        # Create new normalized column
        X['average_charge_normal'] = ((X['average_charge'] - X['average_charge'].min()) /
                            (X['average_charge'].max() - X['average_charge'].min()))

        # Drop old column
        X = X.drop(['average_charge'], axis=1)


        ### tenure : bins
        X['tenure_bins'] = pd.qcut(X['tenure'], q=10, labels=[i for i in range(0,10)]).cat.codes
        X = X.drop(['tenure'], axis=1)
        
        
        ### PhoneService : binarize
        # Get our binary column
        phoneservice_binary = pd.get_dummies(X['PhoneService'], drop_first=True, prefix='PhoneService')
        
        # Concatenate it with our dataframe
        X = pd.concat([X, phoneservice_binary], axis=1)
        
        # and now drop the column that's been processed
        X = X.drop(['PhoneService'], axis=1)

        
        ### Contract : onehotencode
        # get our dummy columns
        contract_dummies = pd.get_dummies(X['Contract'], drop_first=True, prefix='Contract')

        # Concat them with our dataframe
        X = pd.concat([X, contract_dummies], axis=1)

        # and now drop the column
        X = X.drop(['Contract'], axis=1)

        
        # PaymentMethod : onehotencode
        # get our dummy columns
        paymentmethod_dummies = pd.get_dummies(X['PaymentMethod'], drop_first=True, prefix='PaymentMethod', columns=[''])

        # Concat them with our dataframe
        X = pd.concat([X, paymentmethod_dummies], axis=1)

        # and now drop the column
        X = X.drop(['PaymentMethod'], axis=1)

        
        ### MonthlyCharges : normalize
        # Create new normalized column
        X['MonthlyCharges_normal'] = ((X['MonthlyCharges'] - X['MonthlyCharges'].min()) /
                            (X['MonthlyCharges'].max() - X['MonthlyCharges'].min()))

        # Drop old column
        X = X.drop(['MonthlyCharges'], axis=1)

            
        ### TotalCharges : apply log
        
        
        # apply log
        totalcharges_log = X['TotalCharges'].apply(np.log)
        
        # change name before concat
        totalcharges_log = totalcharges_log.rename('TotalCharges_log')
        
        #concat
        X = pd.concat([X, totalcharges_log], axis=1)
        print(type(X)) 
        #drop old
        X = X.drop(['TotalCharges'], axis=1)
                ### Churn : binarize
        
        if 'Churn' in X.columns:
            # Get dummy
            churn_binary = pd.get_dummies(X['Churn'], drop_first=True, prefix='Churn')

            #combine
            X = pd.concat([X, churn_binary], axis=1)

            #drop
            X = X.drop(['Churn'], axis=1)
       
        for rmv in ['PaymentMethod_Mailed check', 'PaymentMethod_Credit card (automatic)', 'PhoneService_Yes']:
            if rmv in X.columns:
                X = X.drop([rmv], axis=1)

        return X
