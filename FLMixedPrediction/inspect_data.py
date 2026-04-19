import pandas as pd
import os

data_dir = '/home/plasmid/Project/FLMixedPrediction/data/Renewable-energy-generation-input-feature-variables-analysis-main/data_processed'

print('=== Wind Farms ===')
wind_dir = os.path.join(data_dir, 'wind_farms')
for file in os.listdir(wind_dir):
    if file.endswith('.xlsx'):
        df = pd.read_excel(os.path.join(wind_dir, file))
        print(f'\n{file}: {len(df.columns)} columns')
        print(f'Columns: {list(df.columns)}')
        print('-'*50)

print('\n=== Solar Stations ===')
solar_dir = os.path.join(data_dir, 'solar_stations')
for file in os.listdir(solar_dir):
    if file.endswith('.xlsx'):
        df = pd.read_excel(os.path.join(solar_dir, file))
        print(f'\n{file}: {len(df.columns)} columns')
        print(f'Columns: {list(df.columns)}')
        print('-'*50)