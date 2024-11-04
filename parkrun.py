import os
import random
import pandas as pd
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import numpy as np

def csv_filename(id) -> str:
    return f"parkrunner_{id}.csv"

def hms_to_minutes(hms: str, sep = ":") -> float:
    parts = hms.split(sep)
    match len(parts):
        case 2:
            # mm:ss
            return int(parts[0]) + int(parts[1])/60
        case 3:
            # hh:mm:ss
            return int(parts[0])*60 + int(parts[1]) + int(parts[2])/60
        case _:
            raise ValueError(f"invalid time '{hms}', should be either hh:mm:ss or mm:ss")

def expected_curve(n, num_samples=1000):
    probs = (n - np.arange(0, float(n))) / n * np.ones((num_samples, n))
    samples = np.random.uniform(0, 1, size=(num_samples, n)) < probs
    runs = np.cumsum(samples, axis=1)
    return np.mean(runs, axis=0)

def scrape_parkrun_results(url, id):
    """
    Scrapes parkrun results from a runner's profile page
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the results table
        # table = [c.text.strip() for c in soup.find_all('caption')]
        table = soup.find(lambda tag: tag.string and tag.string.strip() == 'All  Results').parent
        if not table:
            raise ValueError("No results table found on the page")
        
        # Print headers to debug
        headers = [th.text.strip() for th in table.find_all('th')]
        print("Found headers:", headers)
        
        # Extract rows
        rows = []
        for tr in table.find_all('tr')[1:]:  # Skip header row
            row = []
            for td in tr.find_all('td'):
                row.append(td.text.strip())
            if row:  # Only append non-empty rows
                rows.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(rows, columns=headers)
        
        # Print column names to debug
        print("DataFrame columns:", df.columns.tolist())
        
        # Clean up data types
        date_col = [col for col in df.columns if 'Date' in col][0]  # Find the date column
        event_col = [col for col in df.columns if 'Event' in col][0]  # Find the event column
        time_col = [col for col in df.columns if 'Time' in col][0]  # Find the time column
        
        df[date_col] = pd.to_datetime(df[date_col], format='%d/%m/%Y')
        df['Time_Minutes'] = df[time_col].apply(hms_to_minutes)
        
        return df
    
    except Exception as e:
        print(f"Error: {e}")
        return None

def plot_parkrun_history(df):
    """
    Creates a visualization of parkrun times over time
    """
    plt.style.use('classic')
    
    # Identify column names
    date_col = [col for col in df.columns if 'Date' in col][0]
    event_col = [col for col in df.columns if 'Event' in col][0]
    time_col = [col for col in df.columns if 'Time' in col][0]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[2, 1])
    
    # Color map for different events
    events = df[event_col].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(events)))
    
    # Time series plot
    for event, color in zip(events, colors):
        event_data = df[df[event_col] == event]
        ax1.plot(event_data[date_col], event_data['Time_Minutes'], 
                   label=event, alpha=0.6, color=color)
    
    # Format primary axis
    ax1.set_title('Parkrun Times History', fontsize=14, pad=15)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Time (minutes)')
    
    # Format time labels
    def minutes_formatter(x, p):
        minutes = int(x)
        seconds = int((x - minutes) * 60)
        return f'{minutes}:{seconds:02d}'
    
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(minutes_formatter))
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend with smaller font and move outside plot
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    
    # Rotate x-axis labels
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Event frequency plot
    event_counts = df[event_col].value_counts()
    bars = ax2.bar(range(len(event_counts)), event_counts.values)
    ax2.set_title('Events Frequency')
    ax2.set_xlabel('Event Location')
    ax2.set_ylabel('Number of Runs')
    
    # Set x-tick labels for bar chart
    ax2.set_xticks(range(len(event_counts)))
    ax2.set_xticklabels(event_counts.index, rotation=45, ha='right')
    
    # Color the bars using the same color scheme
    for bar, color in zip(bars, colors[:len(event_counts)]):
        bar.set_color(color)
    
    # Add some statistics as text
    stats_text = (
        f"Total Runs: {len(df)}\n"
        f"Best Time: {minutes_formatter(df['Time_Minutes'].min(), None)}\n"
        f"Average Time: {minutes_formatter(df['Time_Minutes'].mean(), None)}\n"
        f"Most Frequent Event: {df[event_col].mode().iloc[0]}"
    )
    fig.text(0.02, 0.02, stats_text, fontsize=10, family='monospace')
    
    # Adjust layout to prevent overlapping
    plt.tight_layout()
    
    return fig


if __name__ == "__main__":
    id = "3163889"
    url = f"https://www.parkrun.co.za/parkrunner/{id}/all/"
    fname = csv_filename(id)
    try:
        os.stat(fname)
        print("Found CSV, loading...")
        with open(fname) as f:
            df = pd.read_csv(fname)
    except FileNotFoundError:
        df = scrape_parkrun_results(url, id)
        df.to_csv(f"parkrunner_{id}.csv")

    if df is not None:
        # print("\nDataFrame Info:")
        # print(df.info())
        # print("\nFirst few rows:")
        # print(df.head())

        df.sort_values(by="Run Date", ascending=True, inplace=True)
        df.reset_index(inplace=True)
        # df.plot(x="Run Date", y="Time_Minutes")
        
        # fig = plot_parkrun_history(df)
        # plt.show()


        df["avg"] = expected_curve(len(df.index))

        df["seconds"] = df["Time"].astype(str).apply(lambda t: int(t.split(":")[1]))
        df["count"] = df["seconds"].expanding().apply(lambda s: int(s.nunique()))

        # so that they start at 0
        df["count"] -= 1
        df["avg"] -= 1

        print(df['avg'])
        ax = df.plot(y="count", xlim=0, ylim=0)
        df.plot(y="avg", ax=ax)
        print(df)

        seconds = df["seconds"].unique()
        df2 = pd.DataFrame({"second": seconds, "bingo": True}).set_index("second")
        all_seconds = pd.DataFrame({"second": [i for i in range(60)]}).set_index("second").infer_objects()
        df3 = all_seconds.join(df2, on="second")
        df3["bingo"] = df3["bingo"].fillna(False).infer_objects(copy=False)
        df3 = df3.infer_objects()
        print(df3[df3["bingo"] == False])
        print("Num remaining:", len(df3[df3["bingo"] == False].index))

        plt.show()
