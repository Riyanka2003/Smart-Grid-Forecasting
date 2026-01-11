import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_smart_grid_data():
    print("Generating synthetic Smart Grid data...")
    
    # 1. Create a Time Range (1 Year of Hourly Data)
    date_range = pd.date_range(start='2024-01-01', end='2024-12-31 23:00:00', freq='H')
    df = pd.DataFrame(date_range, columns=['Timestamp'])
    
    # 2. Simulate Base Load (The constant hum of the city)
    # Using a sine wave to simulate daily cycles (24 hours)
    # + another sine wave for yearly seasons
    
    x = np.arange(len(df))
    
    # Daily Pattern: Peak at 7 PM (19:00), Low at 3 AM
    daily_seasonality = 10 * np.sin(2 * np.pi * x / 24)
    
    # Yearly Pattern: Peak in Summer (AC usage)
    yearly_seasonality = 20 * np.sin(2 * np.pi * x / (24 * 365))
    
    # Weekly Pattern: Weekends are lower load
    # (day of week 5 and 6 are Sat/Sun)
    weekend_effect = df['Timestamp'].dt.dayofweek.apply(lambda d: -5 if d >= 5 else 0)
    
    # Random Noise (Weather changes, TV spikes, etc.)
    noise = np.random.normal(0, 2, len(df))
    
    # 3. Combine into Total Load (MW)
    base_load = 50 # Minimum 50 MW base
    df['Energy_Consumption_MW'] = base_load + daily_seasonality + yearly_seasonality + weekend_effect + noise
    
    # Ensure no negative values (Physics!)
    df['Energy_Consumption_MW'] = df['Energy_Consumption_MW'].clip(lower=10)
    
    # 4. Save to CSV
    df.to_csv('smart_grid_data.csv', index=False)
    print(f"Success! {len(df)} hourly readings saved to 'smart_grid_data.csv'.")
    
    # 5. Quick Visualization of the first week
    plt.figure(figsize=(12, 6))
    plt.plot(df['Timestamp'][:168], df['Energy_Consumption_MW'][:168], label='Power Usage (MW)', color='teal')
    plt.title('First Week of Smart Grid Data (Hourly)')
    plt.xlabel('Time')
    plt.ylabel('Megawatts (MW)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    generate_smart_grid_data()