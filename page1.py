import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import predictive_maintenance_demo as newData
from collections import OrderedDict
import seaborn as sns

# background_color = "#FFF6FF"
background_color = "white"

def load_machine_data():
    # Load the sensor data from the CSV file
    machine_data = pd.read_csv('predictive_maintenance/predictive_maintenance.csv')
    return machine_data

# Function to plot category distribution
def show_category_distribution(data_frame,label_name,colors,exclude=[],figsize=(10, 5),d=[1, 2],bg_color=None):
    label_dict = data_frame[label_name].value_counts().to_dict()
    for e in exclude:
        del label_dict[e]
    label_dict = OrderedDict(sorted(label_dict.items(), key=lambda label: label[1]))
    label_classes= label_dict.keys()
    label_values = label_dict.values()
    explode = (0,)*(data_frame[label_name].nunique() - len(exclude))
    
    fig, ax = plt.subplots(d[0], d[1], facecolor=bg_color, figsize=figsize)
    sns.countplot(data=data_frame,
                  x=label_name,
                  palette=colors,
                  edgecolor="black",
                  hatch="\\",
                  ax=ax[0],
                  linewidth=1,
                  order = data_frame[label_name].value_counts().index)
    ax[0].set_title(f"Distribution of \"{label_name}\"", size=14, fontweight="bold")
    ax[0].set_xlabel("Class", size=10, fontweight="bold")
    ax[0].set_ylabel("Frequency", size=10, fontweight="bold")
    ax[0].tick_params(axis="x", labelsize=9) 
    ax[0].tick_params(axis="y", labelsize=9)
    ax[0].bar_label(ax[0].containers[0], fmt="%.0f", color="black", fontsize=9)
    
    plt.pie(label_values,
            explode=explode,
            labels=label_classes,
            autopct="%1.2f%%",
            shadow=True,
            startangle=90,
            textprops={"fontsize": 8, "fontweight": "bold", "color": "black"},
            wedgeprops={"edgecolor": "black"},
            colors=colors,
            labeldistance=1.1)
    plt.title(f"Distribution \n of \"{label_name}\". {','.join(exclude)} {'Excluded' if len(exclude)>0 else ''}",
              fontweight="bold",
              fontsize=14)
    
    return fig
    
def main():
    st.title("Machine Data Visualization")
    st.write("Welcome to Page !")
    
    # Load the sensor data
    machine_data = load_machine_data()

    # Display the machine data
    st.subheader("Machine Data")
    st.write(machine_data)

    #histograms
    st.subheader("AirTemp & ProcessTemp")
    fig, ax = plt.subplots(2, 2, figsize=(15,10))
    sns.histplot(data=machine_data, x='AirTemp', kde=True, ax=ax[0,0])
    sns.histplot(data=machine_data, x='ProcessTemp', kde=True, ax=ax[0,1])
    sns.boxplot(data=machine_data, x='AirTemp', ax=ax[1,0])
    sns.boxplot(data=machine_data, x='ProcessTemp', ax=ax[1,1])
    st.pyplot(fig)

    st.subheader("RotationalSpeed, Torque & ToolWear")
    fig1, ax1 = plt.subplots(2, 3, figsize=(15,10))
    sns.histplot(data=machine_data, x='RotationalSpeed', kde=True, ax=ax1[0,0])
    sns.histplot(data=machine_data, x='Torque', kde=True, ax=ax1[0,1])
    sns.histplot(data=machine_data, x='ToolWear', kde=True, ax=ax1[0,2])
    sns.boxplot(data=machine_data, x='RotationalSpeed', ax=ax1[1,0])
    sns.boxplot(data=machine_data, x='Torque', ax=ax1[1,1])
    sns.boxplot(data=machine_data, x='ToolWear', ax=ax1[1,2])
    st.pyplot(fig1.figure)

    #category distribution
    st.subheader("Category Distribution for Type")
    colors = ["#845EC2", "#D65DB1", "#FF6F91", "#FF9671", 
             "#FFC75F", "#008F7A", "#F9F871"]
    plt_figure = show_category_distribution(machine_data,"Type",colors,bg_color=background_color)
    st.pyplot(plt_figure)

    st.subheader("Category Distribution for Target")
    plt_figure1 = show_category_distribution(machine_data,"Target",colors,bg_color=background_color)
    st.pyplot(plt_figure1)
    
    st.subheader("Category Distribution for Failure Type")
    plt_figure2 = show_category_distribution(machine_data,"FailureType",colors,exclude=["No Failure"],figsize=(10, 15),d=[2, 1],
                        bg_color=background_color)
    st.pyplot(plt_figure2)
    
    # Bar plot for maintenance requirement
    st.subheader("Maintenance Requirement")
    maintenance_counts = machine_data['Target'].value_counts()
    st.bar_chart(maintenance_counts)

    # Pairplot to visualize relationships between features
    st.subheader("Pairplot")
    fig2 = sns.pairplot(machine_data, hue='Target')
    st.pyplot(fig2)

    # Correlation heatmap
    st.subheader("Correlation heatmap")
    df_numeric = machine_data.drop(['ProductID', 'Type', 'FailureType'], axis=1)
    corr_matrix = df_numeric.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap')

    # Display the heatmap using Streamlit
    st.pyplot(plt)

if __name__ == '__main__':
    main()
