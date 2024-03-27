# Import libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr


# Configuring the page with Title, icon, and layout
st.set_page_config(
    page_title="EDA for Under-Five Mortality",
    page_icon="/home/U5MRIcon.png",
    layout="wide",
    #initial_sidebar_state="collapsed",  # Optional, collapses the sidebar by default
    menu_items={
        'Get Help': 'https://helppage.ethipiau5m.com',
        'Report a Bug': 'https://bugreport.ethipiau5m.com',
        'About': 'https://ethiopiau5m.com',
    },
)

# Custom CSS to adjust spacing
custom_css = """
<style>
    div.stApp {
        margin-top: -90px !important;  /* We can adjust this value as needed */
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Setting the title with Markdown and center-aligning
st.markdown('<h1 style="text-align: center;">Exploring Under-Five Mortality Dynamics in Ethiopia</h1>', unsafe_allow_html=True)

# Defining background color
st.markdown(
    """
    <style>
    body {
        background-color: #f5f5f5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Defining  header color and font
st.markdown(
    """
    <style>
    h1 {
        color: #3498db;  /* Blue color */
        font-family: 'Helvetica', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)


@st.cache_data
def plot_categorical_feature(data, categorical_feature, figsize1=(8, 5), figsize2=(6, 6), figsize3=(12, 8)):
    plt.style.use('fivethirtyeight')
    sns.set(style="whitegrid")
    # Convert the 'edhs_year' column to string
    data['edhs_year'] = data['edhs_year'].astype(str)

    try:
        # Create a figure with a grid layout
        fig = plt.figure(figsize=(figsize1[0] + figsize2[0], max([figsize1[1], figsize2[1]])))
        gs = gridspec.GridSpec(1, 2, width_ratios=[figsize1[0], figsize2[0]])


        # First Subplot
        ax0 = plt.subplot(gs[0])
        ax0.set_title(f'Distribution by {categorical_feature}')
        sns.countplot(data=data, x=categorical_feature, order=data[categorical_feature].value_counts().index, ax=ax0)
        ax0.set_xlabel(categorical_feature)
        ax0.set_ylabel("Count")
        ax0.tick_params(axis='x', rotation=80)

        # Add labels on top of the bars
        for p in ax0.patches:
            ax0.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', fontsize=8, color='black')

        # Second(pie chart) Subplot
        ax2 = plt.subplot(gs[1])
        ax2.set_title(f'Distribution by {categorical_feature}')
        data[categorical_feature].value_counts().plot.pie(autopct='%1.1f%%',  shadow=True, ax=ax2)

        # Another Bar chart
        fig2, ax1 = plt.subplots(figsize=figsize3)
        ax1.set_title(f'Distribution by {categorical_feature} and EDHS year')
        sns.countplot(data=data, x=categorical_feature, hue='edhs_year', order=data[categorical_feature].value_counts().index, ax=ax1)
        ax1.set_xlabel(categorical_feature)
        ax1.set_ylabel("Count")
        ax1.tick_params(axis='x', rotation=80)

        # Add labels on top of the bars
        for p in ax1.patches:
            ax1.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', fontsize=8, color='black')

        # Show the figures
        st.write(fig)
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"An error occurred while plotting: {e}")


# Function to display frequency distribution table
@st.cache_data
def display_frequency_distribution(dataset, categorical_feature):
    st.write('===================================================================')
    st.write(f'     Percentage distribution of {categorical_feature} feature category based on different EDHS year')
    st.write('===================================================================')

    result = pd.crosstab(
        index=dataset[categorical_feature],
        columns=dataset['edhs_year'],
        values=dataset['child_alive'],
        aggfunc='count',
        margins=True,  # Added 'margins' for total row/column
        margins_name='Total'  # Custom name for the 'margins' column/row
    )
    result['|'] = '|'

    for year in result.columns[:6]:  # Exclude the last column 'Total'
        result[f'{year}(%)'] = (result[year] / result[year]['Total']) * 100

    # Round the percentage values to 1 decimal place
    result = result.round(1).fillna(0)

    # Display the table using st.dataframe
    st.dataframe(result)
    st.write('====================================================================')

    ######################## Correlation Analaysis ##############################################
    # Function to generate the correlation analysis
@st.cache_data
def correlation_analysis(dataset, selected_features, selected_years):
    # Filter dataset based on selected EDHS years
    selected_data = dataset[dataset['edhs_year'].isin(selected_years)]

    # Filter dataset based on selected numerical features
    selected_data = selected_data[selected_features]

    # Calculate the correlation matrix
    corr = selected_data.corr()

    # Calculate p-values
    p_values = pd.DataFrame(index=corr.index, columns=corr.columns)

    for i in range(len(corr)):
        for j in range(len(corr.columns)):
            coef, p_value = pearsonr(selected_data.iloc[:, i], selected_data.iloc[:, j])
            p_values.iloc[i, j] = p_value

    # Round correlation matrix to three decimal places
    corr = corr.round(3)

    # Increase the figure size
    plt.figure(figsize=(16, 12))

    # Plot the correlation heatmap
    sns.heatmap(corr, fmt=".3f", cmap='Blues', cbar_kws={'shrink': 0.8})

    # Manually add text annotations for both correlation and p-values
    for i in range(len(corr)):
        for j in range(len(corr.columns)):
            text = plt.text(j + 0.5, i + 0.5, f"{corr.iloc[i, j]:.3f}\n(p={p_values.iloc[i, j]:.3f})",
                            ha='center', va='center', color='black', fontsize=10)

    plt.title(f"Correlation Plot of selected Features for EDHS Years {', '.join(map(str, selected_years))}")
    st.pyplot(plt)
#########################################################################



#########################################################################

# Additional styling for the overview section
st.markdown(
    """
    ## Welcome to Under-Five Mortality EDA Tool!

    Explore the dynamics of under-five mortality in Ethiopia across different EDHS years. This interactive tool, powered by exploratory data analysis (EDA), offers insights into key trends.

    ### What You Can Do:
    1. Feature Distribution: Examine feature distributions on both aggregated and by specific EDHS year.
    2. Correlation Analysis: Understand feature correlations for any EDHS year.

    Dive into the rich data of EDHS from 2000 to 2019, interact, and uncover valuable insights!
    """
)

# Load cleaned data
cleaned_data_path = '/home/datasetProcessed.csv'
dataset = load_data(cleaned_data_path)

# List of features to exclude from the dropdown menu
excluded_features = ['time_taken_water_source', 'age_household_head', 'latitude', 'longitude', 'altitude_in_meter', 'u5child', 'u5child_mortality']

# Filter out excluded features
allowed_features = [col for col in dataset.columns if col not in excluded_features]

# Function to create a horizontal line with custom styling
def horizontal_line(height=1, color="blue", margin="0.5em 0"): # color="#ddd"
    return f'<hr style="height: {height}px; margin: {margin}; background-color: {color};">'

# Sidebar for selecting parameters
st.sidebar.header('Parameters')


####################### 1.Feature Distribution of UFM ##################################
st.sidebar.markdown('### Feature Distribution')
# Allow the user to select a feature
selected_feature = st.sidebar.selectbox('Select Feature', allowed_features)

# Display distribution plots and tables based on the selected feature
if st.sidebar.button('Show Distribution'):
    st.subheader(f'Frequency Distribution of {selected_feature}')
    display_frequency_distribution(dataset, selected_feature)
    st.subheader(f'Distribution of {selected_feature}')
    plot_categorical_feature(dataset, selected_feature)


# Separator
st.sidebar.markdown(horizontal_line(), unsafe_allow_html=True)


####################### 2. Correlation Analysis ###################################################
st.sidebar.header('Correlation Analysis')
# Filter numerical features only
numerical_features = dataset.select_dtypes(include=['number']).columns

# Allow the user to select the features
selected_features = st.sidebar.multiselect('Select Features for Correlation', numerical_features, key='features')

# Allow the user to select multiple EDHS years using a multiselect
selected_years = st.sidebar.multiselect('Select EDHS Years', dataset['edhs_year'].unique(), key='years')

# Button to generate correlation matrix
if st.sidebar.button('Generate Correlation Matrix'):
    correlation_analysis(dataset, selected_features, selected_years)

# Separator
st.sidebar.markdown(horizontal_line(), unsafe_allow_html=True)