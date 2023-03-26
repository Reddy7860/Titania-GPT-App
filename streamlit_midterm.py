import streamlit as st
import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay, CustomBusinessMonthEnd
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
from matplotlib.dates import date2num
import mpl_finance
import matplotlib.dates as mdates
import altair as alt
import datetime as dt

# Load dataset
df = pd.read_csv("Titania_visual.csv")

# Define available filters/inputs
strategies = sorted(df['Strategy'].astype(str).unique())

strategies = [s for s in strategies if str(s) != 'nan' and s != 'Select All']

strategies.insert(0, "Select All")  # Add "Select All" option to strategy filter


# Sidebar filters/inputs
st.sidebar.title("Filter Data")
selected_strategy = st.sidebar.selectbox("Select a Strategy", strategies, index=strategies.index("Reds Brahmos"))

# Get unique dates in filtered dataframe
filtered_df = df[df['Strategy'] == selected_strategy]
dates = sorted(filtered_df['Date'].unique(), reverse=True)

# Select date
selected_date = st.selectbox("Select a Date", dates)

filtered_df['Datetime'] = pd.to_datetime(filtered_df['Datetime'])
df['Datetime'] = pd.to_datetime(df['Datetime'])

# Filter dataframe for selected date
selected_datetime = pd.to_datetime(selected_date)
candle_df = df[df['Datetime'].dt.date == selected_datetime.date()]
candle_df['Datetime'] = candle_df['Datetime'].dt.tz_localize(None)

# Filter dataframe for selected date
filtered_df = filtered_df[filtered_df['Date'] == selected_date]

# Calculate candlestick colors
filtered_df['color'] = ['green' if close >= open else 'red' for open, close in zip(filtered_df['Open'], filtered_df['Close'])]

# Calculate cumulative gains
filtered_df['cumulative_gains'] = filtered_df['Profit/Loss'].cumsum()

print(filtered_df)

# Set y-axis range for candlestick chart
y_min = candle_df['Low'].min() - 50
y_max = candle_df['High'].max() + 50



# Convert Datetime column to datetime format
candle_df['Datetime'] = pd.to_datetime(candle_df['Datetime'])

# Subtract 1.5 hours from each datetime value
candle_df['Datetime'] = candle_df['Datetime'] - dt.timedelta(hours=1, minutes=30)

# # Create plot of candlestick and cumulative gains
# candlestick = alt.Chart(candle_df).mark_bar().encode(
#     x='Datetime:T',
#     y=alt.Y('Open', axis=alt.Axis(title='Price'), scale=alt.Scale(domain=[y_min, y_max])),
#     y2='Close',
#     color=alt.condition(
#         alt.datum.color == 'green',
#         alt.value('green'),
#         alt.value('red')
#     )
# ).properties(
#     width=800,
#     height=300
# )

# Create plot of candlestick and cumulative gains
candlestick = alt.Chart(candle_df).mark_line().encode(
    x='Datetime:T',
    y=alt.Y('Open', axis=alt.Axis(title='Price'), scale=alt.Scale(domain=[y_min, y_max])),
    color=alt.condition(
        alt.datum.color == 'green',
        alt.value('green'),
        alt.value('red')
    )
).properties(
    width=800,
    height=300
)

# Create signal highlight chart
highlight_signal = alt.Chart(filtered_df[filtered_df['Signal'] == selected_strategy]).mark_point(
    shape='circle',
    size=100,
    fill='black',
    stroke='black',
    strokeWidth=1
).encode(
    x='Datetime:T',
    y='High:Q'
)

print("filtered_df")
print(filtered_df.columns)
# print(filtered_df[filtered_df['Signal'] == selected_strategy])

new_df = filtered_df.copy()
# Subtract 7 hours from each datetime value
new_df['Datetime'] = new_df['Datetime'] - dt.timedelta(hours=7)

print(new_df.columns)

# # Combine charts
# signal_highlight = alt.Chart(new_df).mark_point(
#     shape='triangle-up',
#     size=100,
#     fill='yellow',
#     stroke='black',
#     strokeWidth=1
# ).encode(
#     x='Datetime:T',
#     y='High:Q',
#     tooltip=['Signal', 'Datetime']
# )

# Combine charts
signal_highlight = alt.Chart(new_df).mark_rule(
    color='orange',
    strokeDash=[5, 5]
).encode(
    y=alt.Y('Value:Q', axis=alt.Axis(title='Value')),
    tooltip=['Signal', 'Datetime']
)


st.altair_chart((candlestick + signal_highlight).properties(
title={
"text": [f"Candlestick and Cumulative Gains for {selected_strategy} Strategy ({selected_date})"],
"subtitle": ["Price vs. Time"]
}
).configure_axis(
labelFontSize=12,
titleFontSize=14
).configure_title(
fontSize=16
).properties(
height=500,
width=800
).interactive(), use_container_width=True)

datetimes = sorted(filtered_df['Datetime'].unique(), reverse=True)


if len(datetimes) > 0:
	st.write(f"List of signals for {selected_strategy} strategy:")
	for datetime in datetimes:
		signal = filtered_df[filtered_df['Datetime'] == datetime]['Signal'].iloc[0]
		buy_probability = filtered_df[filtered_df['Datetime'] == datetime]['buy_probability'].iloc[0]
		sell_probability = filtered_df[filtered_df['Datetime'] == datetime]['sell_probability'].iloc[0]
		value = filtered_df[filtered_df['Datetime'] == datetime]['Value'].iloc[0]
		st.write(f"- {datetime}: {signal} {buy_probability} {sell_probability}{value}")
		if signal == selected_strategy:
			signal_highlight_chart = alt.Chart(pd.DataFrame({'Datetime': [datetime]})).mark_point(
			shape='triangle-up',
			size=100,
			fill='yellow',
			stroke='black',
			strokeWidth=1
			).encode(
			x='Datetime:T',
			y='High:Q',
			tooltip=['Signal','buy_probability','sell_probability']
			)
			st.altair_chart(signal_highlight_chart, use_container_width=True)
		else:
			st.write("No data available for selected filters.")

# stocks = sorted(df['Stock'].astype(str).unique())
# indicators = ['SMA_Call', 'RSI_Call', 'MACD_Call', 'Pivot_Call', 'PCR_Call', 'BB_Call', 'VWAP_Call', 'SuperTrend_Call']
# dates = sorted(df['Date'].unique())

# # Sidebar filters/inputs
# st.sidebar.title("Filter Data")
# selected_stock = st.sidebar.selectbox("Select a Stock", stocks)
# selected_indicator = st.sidebar.selectbox("Select an Indicator", indicators)
# selected_start_date = st.sidebar.selectbox("Select a Start Date", dates)
# selected_end_date = st.sidebar.selectbox("Select an End Date", dates)

# # Apply filters/inputs to the dataset
# filtered_df = df[(df['Stock'] == selected_stock) & (df['Date'] >= selected_start_date) & (df['Date'] <= selected_end_date)]

# # Graph 1: Line chart of indicator values over time
# fig1 = px.line(filtered_df, x='Date', y=selected_indicator, title=f"{selected_indicator} for {selected_stock}")
# st.plotly_chart(fig1)

# # Graph 2: Bar chart of buy and sell probabilities by day
# grouped_df = filtered_df.groupby(['Date']).agg({'buy_probability': 'mean', 'sell_probability': 'mean'}).reset_index()
# fig2 = px.bar(grouped_df, x='Date', y=['buy_probability', 'sell_probability'], barmode='group', title="Buy/Sell Probabilities by Day")
# st.plotly_chart(fig2)


# # Define available filters/inputs
# strategies = sorted(df['Strategy'].astype(str).unique())

# strategies.insert(0, "Select All")  # Add "Select All" option to strategy filter

# # Sidebar filters/inputs
# st.sidebar.title("Filter Data")
# selected_strategy = st.sidebar.selectbox("Select a Strategy", strategies)

# df['Datetime'] = pd.to_datetime(df['Datetime'])

# # Create date range filters
# min_date = pd.to_datetime(df['Datetime'].min()).date()
# max_date = pd.to_datetime(df['Datetime'].max()).date()

# selected_start_date = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date, key='start')
# selected_end_date = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date, key='end')

# # Create custom business day calendar to exclude weekends
# bday = CustomBusinessDay(calendar={'weekday': True, 'weekend': False})

# # Apply filters/inputs to the dataset
# if selected_strategy == "Select All":
#     filtered_df = df[(df['Datetime'].dt.date >= selected_start_date) & (df['Datetime'].dt.date <= selected_end_date)]
# else:
#     filtered_df = df[(df['Strategy'] == selected_strategy) & (df['Datetime'].dt.date >= selected_start_date) & (df['Datetime'].dt.date <= selected_end_date)]

# # Calculate candlestick colors
# colors = []
# for i in range(len(filtered_df)):
#     if filtered_df.iloc[i]['Close'] >= filtered_df.iloc[i]['Open']:
#         colors.append('green')
#     else:
#         colors.append('red')

# # Calculate cumulative gains
# cumulative_gains = filtered_df['Profit/Loss'].cumsum()

# # Create plot of candlestick and cumulative gains
# fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)

# # Create candlestick trace
# fig.add_trace(go.Candlestick(
#     x=filtered_df.index,
#     open=filtered_df['Open'],
#     high=filtered_df['High'],
#     low=filtered_df['Low'],
#     close=filtered_df['Close'],
#     increasing=dict(line=dict(color='green')),
#     decreasing=dict(line=dict(color='red')),
#     showlegend=False,
#     name='Candlestick'
# ), row=1, col=1)

# # Create cumulative gains trace
# fig.add_trace(go.Scatter(
#     x=filtered_df.index,
#     y=cumulative_gains,
#     name='Cumulative Gains',
#     mode='lines',
#     line=dict(color='blue')
# ), row=2, col=1)

# # Set layout
# fig.update_layout(
#     title=f"Candlestick and Cumulative Gains for {selected_strategy} Strategy",
#     xaxis_title="Datetime",
#     height=800,
#     template='plotly_dark'
# )

# # Set y-axis range for candlestick chart
# fig.update_yaxes(
#     range=[filtered_df['Low'].min() - 100, filtered_df['High'].max() + 100],
#     row=1,
#     col=1
# )

# # Set y-axis range for cumulative gains chart
# fig.update_yaxes(
#     range=[0, cumulative_gains.max()],
#     row=2,
#     col=1
# )

# # Show plot
# st.plotly_chart(fig)


# # Define available filters/inputs
# strategies = sorted(df['Strategy'].astype(str).unique())

# strategies.insert(0, "Select All")  # Add "Select All" option to strategy filter

# # Sidebar filters/inputs
# st.sidebar.title("Filter Data")
# selected_strategy = st.sidebar.selectbox("Select a Strategy", strategies)

# # Filter data by selected strategy
# if selected_strategy == "Select All":
#     filtered_df = df
# else:
#     filtered_df = df[df['Strategy'] == selected_strategy]

# # Get unique dates in filtered dataframe
# dates = sorted(filtered_df['Date'].unique())

# # Select date
# selected_date = st.selectbox("Select a Date", dates)

# filtered_df['Datetime'] = pd.to_datetime(filtered_df['Datetime'])
# df['Datetime'] = pd.to_datetime(df['Datetime'])

# # Filter dataframe for selected date
# selected_datetime = pd.to_datetime(selected_date)
# candle_df = df[df['Datetime'].dt.date == selected_datetime.date()]
# candle_df['Datetime'] = candle_df['Datetime'].dt.tz_localize(None)

# # Filter dataframe for selected date
# filtered_df = filtered_df[filtered_df['Date'] == selected_date]
# print(selected_date)
# print(filtered_df.head())

# # Calculate candlestick colors
# colors = []
# for i in range(len(filtered_df)):
#     if filtered_df.iloc[i]['Close'] >= filtered_df.iloc[i]['Open']:
#         colors.append('green')
#     else:
#         colors.append('red')

# # Calculate cumulative gains
# cumulative_gains = filtered_df['Profit/Loss'].cumsum()

# # Create plot of candlestick and cumulative gains
# fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)

# # Create candlestick trace
# fig.add_trace(go.Candlestick(
#     x=candle_df['Datetime'],
#     open=candle_df['Open'],
#     high=candle_df['High'],
#     low=candle_df['Low'],
#     close=candle_df['Close'],
#     increasing=dict(line=dict(color='green')),
#     decreasing=dict(line=dict(color='red')),
#     showlegend=False,
#     name='Candlestick'
# ), row=1, col=1)

# # Create signal highlight trace
# signals = filtered_df[filtered_df['Signal'] == 'Gap_Up']
# if len(signals) > 0:
#     fig.add_trace(go.Scatter(
#         x=signals['Datetime'],
#         y=signals['High'],
#         mode='markers',
#         marker=dict(symbol='triangle-up', size=10, color='yellow', line=dict(color='black', width=1)),
#         name='Signal Highlight'
#     ), row=1, col=1)

# # Create cumulative gains trace
# fig.add_trace(go.Scatter(
#     x=filtered_df['Datetime'],
#     y=cumulative_gains,
#     name='Cumulative Gains',
#     mode='lines',
#     line=dict(color='blue')
# ), row=2, col=1)

# # Set layout
# fig.update_layout(
#     title=f"Candlestick and Cumulative Gains for {selected_strategy} Strategy ({selected_date})",
#     xaxis_title="Datetime",
#     height=800,
#     template='plotly_dark'
# )

# # Set y-axis range for candlestick chart
# fig.update_yaxes(
#     range=[filtered_df['Low'].min() - 100, filtered_df['High'].max() + 100],
#     row=1,
#     col=1
# )

# # Set y-axis range for cumulative gains chart
# # fig.update_yaxes(
# #     range=[0, cumulative_gains.max()],
# #     row=2,
# #     col=1
# # )

# # Show plot
# st.plotly_chart(fig)

# # Display list of dates and signal for selected strategy
# dates = sorted(filtered_df['Datetime'].dt.date.unique(), reverse=True)
# if len(dates) > 0:
# 	st.write(f"List of dates and signals for {selected_strategy} strategy:")
# 	for date in dates:
# 		st.write(f"- {date}: {filtered_df[filtered_df['Datetime'].dt.date == date]['Signal'].iloc[0]}")
# else:
# 	st.write("No data available for selected filters.")

# # Define available filters/inputs
# strategies = sorted(df['Strategy'].astype(str).unique())

# strategies.insert(0, "Select All")  # Add "Select All" option to strategy filter

# # Sidebar filters/inputs
# st.sidebar.title("Filter Data")
# selected_strategy = st.sidebar.selectbox("Select a Strategy", strategies)

# # Filter data by selected strategy
# if selected_strategy == "Select All":
#     filtered_df = df
# else:
#     filtered_df = df[df['Strategy'] == selected_strategy]

# # Get unique dates in filtered dataframe
# dates = sorted(filtered_df['Date'].unique())

# # Select date
# selected_date = st.selectbox("Select a Date", dates)

# filtered_df['Datetime'] = pd.to_datetime(filtered_df['Datetime'])
# df['Datetime'] = pd.to_datetime(df['Datetime'])

# # Filter dataframe for selected date
# selected_datetime = pd.to_datetime(selected_date)
# candle_df = df[df['Datetime'].dt.date == selected_datetime.date()]
# candle_df['Datetime'] = candle_df['Datetime'].dt.tz_localize(None)

# # Filter dataframe for selected date
# filtered_df = filtered_df[filtered_df['Date'] == selected_date]

# # Convert datetime to matplotlib format
# candle_df['mpl_date'] = candle_df['Datetime'].apply(lambda date: mdates.date2num(date))

# # Calculate candlestick colors
# colors = []
# for i in range(len(filtered_df)):
#     if filtered_df.iloc[i]['Close'] >= filtered_df.iloc[i]['Open']:
#         colors.append('green')
#     else:
#         colors.append('red')

# # Calculate cumulative gains
# cumulative_gains = filtered_df['Profit/Loss'].cumsum()

# # Create plot of candlestick and cumulative gains
# fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(12, 8))

# # Create candlestick trace
# candlestick_ohlc(ax1, candle_df[['mpl_date', 'Open', 'High', 'Low', 'Close']].values, width=0.01, colorup='g', colordown='r')
# ax1.set_title(f"Candlestick Chart for {selected_strategy} Strategy ({selected_date})")

# # Create signal highlight trace
# signals = filtered_df[filtered_df['Signal'] == 'Gap_Up']
# if len(signals) > 0:
#     signal_dates = signals['Datetime'].apply(lambda date: mdates.date2num(date))
#     ax1.scatter(signal_dates, signals['High'], color='yellow', marker='^', s=100, label='Signal Highlight')
#     ax1.legend()

# # Create cumulative gains trace
# ax2.plot(filtered_df['Datetime'], cumulative_gains, color='blue')
# ax2.set_title(f"Cumulative Gains for {selected_strategy} Strategy ({selected_date})")

# # Set x-axis label
# fig.text(0.5, 0.04, 'Datetime', ha='center')

# # Set y-axis label
# fig.text(0.04, 0.5, 'Price', va='center', rotation='vertical')

# # Show plot
# st.pyplot(fig)


# # Define available filters/inputs
# strategies = sorted(df['Strategy'].astype(str).unique())

# strategies.insert(0, "Select All")  # Add "Select All" option to strategy filter

# # Sidebar filters/inputs
# st.sidebar.title("Filter Data")
# selected_strategy = st.sidebar.selectbox("Select a Strategy", strategies)

# # Filter data by selected strategy
# if selected_strategy == "Select All":
#     filtered_df = df
# else:
#     filtered_df = df[df['Strategy'] == selected_strategy]

# # Get unique dates in filtered dataframe
# dates = sorted(filtered_df['Date'].unique())

# # Select date
# selected_date = st.selectbox("Select a Date", dates)

# filtered_df['Datetime'] = pd.to_datetime(filtered_df['Datetime'])
# df['Datetime'] = pd.to_datetime(df['Datetime'])

# # Filter dataframe for selected date
# selected_datetime = pd.to_datetime(selected_date)
# candle_df = df[df['Datetime'].dt.date == selected_datetime.date()]
# candle_df['Datetime'] = candle_df['Datetime'].dt.tz_localize(None)

# # Filter dataframe for selected date
# filtered_df = filtered_df[filtered_df['Date'] == selected_date]

# # Calculate candlestick colors
# filtered_df['color'] = ['green' if close >= open else 'red' for open, close in zip(filtered_df['Open'], filtered_df['Close'])]

# # Calculate cumulative gains
# filtered_df['cumulative_gains'] = filtered_df['Profit/Loss'].cumsum()

# # Set y-axis range for candlestick chart
# y_min = candle_df['Low'].min() - 50
# y_max = candle_df['High'].max() + 50

# # Create plot of candlestick and cumulative gains
# candlestick = alt.Chart(candle_df).mark_bar().encode(
#     x='Datetime:T',
#     y=alt.Y('Open', axis=alt.Axis(title='Price'), scale=alt.Scale(domain=[y_min, y_max])),
#     y2='Close',
#     color=alt.condition(
#         alt.datum.color == 'green',
#         alt.value('green'),
#         alt.value('red')
#     )
# ).properties(
#     width=800,
#     height=300
# )


# signal_highlight = alt.Chart(filtered_df[filtered_df['Signal'] == selected_strategy]).mark_point(
#     shape='triangle-up',
#     size=100,
#     fill='yellow',
#     stroke='black',
#     strokeWidth=1
# ).encode(
#     x='Datetime:T',
#     y='High:Q',
#     tooltip=['Signal']
# )

# cumulative_gains = alt.Chart(filtered_df).mark_line().encode(
#     x='Datetime:T',
#     y=alt.Y('cumulative_gains:Q', axis=alt.Axis(title='Cumulative Gains')),
#     color=alt.value('blue')
# ).properties(
#     width=800,
#     height=100
# )



# # Show plot
# st.altair_chart((candlestick + signal_highlight + cumulative_gains).properties(
#     title={
#         "text": [f"Candlestick and Cumulative Gains for {selected_strategy} Strategy ({selected_date})"],
#         "subtitle": ["Price vs. Time"]
#     }
# ).configure_axis(
#     labelFontSize=12,
#     titleFontSize=14
# ).configure_title(
#     fontSize=16
# ).properties(
#     height=500,
#     width=800
# ).interactive(), use_container_width=True)

# # Display list of dates and signal for selected strategy
# dates = sorted(filtered_df['Datetime'].dt.date.unique(), reverse=True)
# if len(dates) > 0:
# 	st.write(f"List of dates and signals for {selected_strategy} strategy:")
# 	for date in dates:
# 		st.write(f"- {date}: {filtered_df[filtered_df['Datetime'].dt.date == date]['Signal'].iloc[0]}")
# else:
# 	st.write("No data available for selected filters.")