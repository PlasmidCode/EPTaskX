#!/usr/bin/env python3
"""
Generate synthetic PV and wind power generation data for federated learning experiments.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Generate timestamp index
timestamps = pd.date_range(start='2023-01-01', end='2023-12-31', freq='H')

# Function to generate synthetic PV data
def generate_pv_data(timestamps):
    """Generate synthetic PV power data with temperature and irradiance features"""
    # Basic sine wave for daily cycle
    daily_cycle = np.sin(2 * np.pi * (timestamps.hour + timestamps.minute/60) / 24)
    daily_cycle = np.maximum(daily_cycle, 0)  # Only positive values during day
    
    # Seasonal variation
    seasonal = np.sin(2 * np.pi * (timestamps.dayofyear) / 365)
    seasonal = (seasonal + 1) / 2  # Normalize to 0-1
    
    # Random fluctuations
    noise = np.random.normal(0, 0.1, len(timestamps))
    
    # Temperature (correlated with solar irradiance)
    temperature = 15 + 10 * daily_cycle + 5 * seasonal + np.random.normal(0, 2, len(timestamps))
    
    # Irradiance
    irradiance = 1000 * daily_cycle * seasonal + np.random.normal(0, 50, len(timestamps))
    irradiance = np.maximum(irradiance, 0)
    
    # PV power (W/m2 to kW for a 10kW system)
    power = 10 * (irradiance / 1000) * (0.15 + 0.05 * np.random.randn(len(timestamps)))
    power = np.maximum(power, 0)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'temperature': temperature,
        'irradiance': irradiance,
        'power': power
    })
    
    return df

# Function to generate synthetic wind data
def generate_wind_data(timestamps):
    """Generate synthetic wind power data with wind speed and direction features"""
    # Wind speed (more variable than PV)
    wind_speed = np.abs(np.random.normal(5, 3, len(timestamps)))
    
    # Wind direction (0-360 degrees)
    wind_direction = np.random.uniform(0, 360, len(timestamps))
    
    # Wind power follows a cubic relationship with wind speed
    # Cut-in speed: 3 m/s, rated speed: 12 m/s, cut-out speed: 25 m/s
    power = np.zeros(len(timestamps))
    
    # Below cut-in: 0 power
    below_cutin = wind_speed < 3
    above_cutout = wind_speed > 25
    
    # Between cut-in and rated: cubic relationship
    between = ~below_cutin & ~above_cutout
    power[between] = 10 * (wind_speed[between]**3 / 12**3)
    
    # Above rated but below cut-out: rated power
    above_rated = (wind_speed >= 12) & ~above_cutout
    power[above_rated] = 10
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'wind_speed': wind_speed,
        'wind_direction': wind_direction,
        'power': power
    })
    
    return df

# Generate data
pv_data = generate_pv_data(timestamps)
wind_data = generate_wind_data(timestamps)

# Save to CSV files
pv_data.to_csv('data/pv_site.csv', index=False)
wind_data.to_csv('data/wind_site.csv', index=False)

print(f"Generated PV data: {len(pv_data)} records")
print(f"Generated wind data: {len(wind_data)} records")
print(f"Data saved to data/pv_site.csv and data/wind_site.csv")