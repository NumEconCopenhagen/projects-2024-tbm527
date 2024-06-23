import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

def plot_entire_data_set(yearly_pricedata):
    # We set up the figure and axis
    plt.figure(figsize=(14, 8))

    # We loop through each unique area in the DataFrame and plot it
    for area in yearly_pricedata['Area'].unique():
        area_data = yearly_pricedata[yearly_pricedata['Area'] == area]
        plt.plot(area_data['Year'], area_data['Price'], marker='o', label=area)

    # We add title and labels
    plt.title('Annual Condominium Prices Across All Regions')
    plt.xlabel('Year')
    plt.ylabel('Average Condominium Price (1000 Krones)')

    # We add a legend to the plot
    plt.legend(title='Area', bbox_to_anchor=(1.05, 1), loc='upper left')

    # We rotate the x-axis labels for better readability
    plt.xticks(rotation=45)

    # We add grid lines
    plt.grid(True)

    # We adjust layout to make room for the legend
    plt.tight_layout()

    # We show the plot
    plt.show()

def plot_annual_prices_area(areas_of_interest, focused_pricedata):
    # We'll create a line plot for each area
    for area in areas_of_interest:
        # We filter the data for the current area
        area_data = focused_pricedata[focused_pricedata['Area'] == area]
        
        # We plot the data
        plt.figure(figsize=(12, 6))  # Set the figure size
        plt.plot(area_data['Year'], area_data['Price'], marker='o', label=area)  # Plot the data
        
        # We add title and labels
        plt.title(f'Annual Condominium Prices in {area}')
        plt.xlabel('Year')
        plt.ylabel('Average Condominium Price (in 1000 DKK)')
        plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
        plt.legend()  # Show the legend
        plt.grid(True)  # Show grid lines for better readability
        plt.tight_layout()  # Adjust the layout

        # We show the plot
        plt.show()

def plot_annual_sales_area(areas_of_interest, focused_salesdata):
    # We'll create a line plot for each area
    for area in areas_of_interest:
        # We filter the data for the current area
        area_data = focused_salesdata[focused_salesdata['Area'] == area]
        
        # We plot the data
        plt.figure(figsize=(12, 6))  # Set the figure size
        plt.plot(area_data['Year'], area_data['Sales'], marker='o', label=area)  # Plot the data
        
        # We add title and labels
        plt.title(f'Annual Number of Condominium Sales in {area}')
        plt.xlabel('Year')
        plt.ylabel('Total Number of Condominium sales')
        plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
        plt.legend()  # Show the legend
        plt.grid(True)  # Show grid lines for better readability
        plt.tight_layout()  # Adjust the layout

        # We show the plot
        plt.show()

def plot_combined_prices(focused_pricedata):
        fig = px.line(focused_pricedata, x='Year', y='Price', color='Area',
                title='Annual Condominium Prices',
                labels={'Price': 'Average Condominium Price (1000 Krones)'})
        fig.update_xaxes(tickangle=45)
        fig.show()

def plot_combined_sales(focused_salesdata):
    fig = px.line(focused_salesdata, x='Year', y='Sales', color='Area',
                title='Annual Condominium Sales',
                labels={'Sales': 'Total Condominium Sales'})
    fig.update_xaxes(tickangle=45)
    fig.show()

def plot_merged_data(areas_of_interest, merged_data):
    # We plot our merged data
    for area in areas_of_interest:
        # We filter the data for the current area
        area_data = merged_data[merged_data['Area'] == area]
        
        # We create a new figure and a twin Y-axis for the second variable
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # We plot the 'Price' on the primary y-axis
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Average Condominium Price (1000 Krones)', color='tab:blue')
        ax1.plot(area_data['Year'], area_data['Price'], color='tab:blue', marker='o', label='Price')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        # We create a second y-axis for the 'Sales' data
        ax2 = ax1.twinx()
        ax2.set_ylabel('Number of Sales', color='tab:red')
        ax2.plot(area_data['Year'], area_data['Sales'], color='tab:red', marker='o', label='Sales')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        # Wotate the x-axis labels for better readability
        ax1.set_xticklabels(area_data['Year'], rotation=45)

        # We add a title and a grid
        plt.title(f'Annual Condominium Prices and Number of Sales in {area}')
        ax1.grid(True)

        # We show the plot with a tight layout
        fig.tight_layout()
        plt.show()