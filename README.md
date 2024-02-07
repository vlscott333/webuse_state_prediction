# webuse_state_prediction

import numpy as np
import pandas as pd
import statsmodels.formula.api as sms

internet = pd.read_csv('internetusage.csv')# load the file internetusage.csv
internetdf= pd.DataFrame(internet)
pd.options.display.max_columns = None
#print(internetdf.head())

#df = df.astype({"a": int, "b": complex})
interbachdf = pd.DataFrame(internet.astype({"bachelors_degree": float, "internet_usage": float, "State": str}))
#print(interbachdf.head())

interbachdf_final = pd.DataFrame(interbachdf[['State', 'bachelors_degree', 'internet_usage']])
#print(interbachdf_final.head())

model = sms.ols('internet_usage ~ bachelors_degree', data= interbachdf).fit()
print(model.summary()) # fit a linear model using the sms.ols function and the internet dataframe

#use the model.predict function to find the predicted value for internet_usage using
bach_percent = float(input('-'*100 +'\n\nEnter a value (i.e. 1.00-100.00) to predict internet usage'
                                    '\nbased on percent of population with a bachelor\'s degree: ' ))
prediction = pd.DataFrame(np.array([[bach_percent]]), columns = ['bachelors_degree'])
parse_prediction = str.split(str(model.predict(prediction)))
predict_only = (str(parse_prediction[1:2]))
predict_only_strp = (predict_only.strip('['']''\''))
print('When a state has a population of ' + str(bach_percent)
      + '% of people with a bachelors degree,\n'
        'then we can predict that approximately '
      + str(predict_only_strp) + '% of the population of '
        'that state use internet.')
# the bach_percent value for the predictor
