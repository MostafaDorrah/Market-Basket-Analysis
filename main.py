import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, fpgrowth, fpmax, association_rules, hmine
from collections import Counter
import altair as alt
import re


st.set_option('deprecation.showPyplotGlobalUse', False)
def cleaner(df):
    df=df.dropna()

    df['Customer ID'] = df['Customer ID'].astype(str).str.replace('\D', '')

    df['Items'] = df['Items'].str.strip()

    df['Country'] = df['Country'].str.strip()

    df['Month'] = df['Month'].apply(lambda x: re.sub(r'\D', '', str(x)))

    df['Price'] = df['Price'].apply(lambda x: re.sub(r'[^\d.]', '', str(x)))

    df["Price"] = df["Price"].astype(float)

    df["Month"] = df["Month"].astype(int)
    return df

# Add logo
st.image('logo.png',width = 200)

# Page header
st.write('# Welcome To :orange[BasketGraphix]:')
st.write('## The place where you can understand and get more insights about your own data')

# Text and requirements
st.write('Please upload your data and make sure it fits the following requirements:')
st.write('1. It must contain the following columns: Month, Items, Quantity, Price, Transaction, Customer ID, Country')
st.write('2. It must be in the format of .csv')

file = st.file_uploader("Upload CSV", type="csv")

if file is not None:

    df = pd.read_csv(file)

    selected_columns = ['Country', 'Customer ID','Price','Month','Quantity','Items','Transaction']
    df = df[selected_columns]

    cleaner(df)
    st.write('#### Here is your data and some insights about it:')
    st.dataframe(df)

    df_count = df.groupby('Country').count().reset_index()
    st.write('### :orange[Customer Distribution by Country]')
    fig = px.choropleth(df_count,
                        locations='Country',
                        locationmode='country names',
                        color='Customer ID',
                        hover_data=['Country', 'Customer ID'],
                        labels={'Customer ID': 'Number of Customers'},
                        color_continuous_scale='Viridis')
    fig.update_layout(height=500, width=2000)

    st.plotly_chart(fig)

    #  #plot 2
    st.write('### :orange[Top 10 Products]')
    top_10_products = df.groupby('Items')['Customer ID'].nunique().nlargest(10)

    fig, ax = plt.subplots(figsize=(7, 3))

    ax.hlines(y=top_10_products.index, xmin=0, xmax=top_10_products, color='skyblue')
    ax.plot(top_10_products, top_10_products.index, "o")
    ax.set_xlabel('Number of Customers')
    ax.set_ylabel('Product Name')
    ax.set_title('Top 10 Products by Sales')

    plt.tight_layout(pad=4)
    st.pyplot(fig)


#plot 3
    st.write('### :orange[Top 10 Customers]')
    mydf = df.groupby('Customer ID')['Transaction'].nunique().reset_index()
    mydf = mydf.sort_values('Transaction', ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax = sns.barplot(x=mydf['Customer ID'], y=mydf['Transaction'])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.tight_layout(pad=3)
    st.pyplot(fig)

# #plot 4
    st.write('### :orange[Sales vs Month]')
    df_2 = df.groupby('Month').sum()
    fig, ax = plt.subplots(figsize=(7, 3))
    sns.lineplot(data=df_2, x='Month', y='Price', ax=ax)
    plt.ylabel('Sales')
    plt.tight_layout(pad=3)
    st.pyplot(fig)


#algorthms

    def func(df):
        df[df>0.0 ] = 1

        return df


    idx = df[df['Transaction'].str.contains('C')].index
    df = df.drop(index=idx, axis=0)
    df = df.dropna()

    mydf = df.groupby(['Transaction', "Items"])['Quantity'].sum().unstack().fillna(0)
    mydf.apply(func)


    #apriori

    opr_itemset = apriori(mydf, min_support=0.02, use_colnames=True)

    opr_itemset["itemsets"] = opr_itemset["itemsets"].apply(lambda x: list(x))

    data = []
    support1 = []



    for i in range(opr_itemset.shape[0]):
        if len(opr_itemset.iloc[i]["itemsets"]) >= 2:
            data.append(opr_itemset.iloc[i]["itemsets"])
            support1.append(opr_itemset.iloc[i]['support'])

    st.write('# In This Part we will Analyze the Data using :orange[Apriori Algorithm] :')
    #graph 1
    st.title('The Apriori Algorithm')
    st.write('### :orange[Item Frequency] :')

    flat_data = [item for sublist in data for item in sublist]
    frequency = dict(Counter(flat_data))
    df = pd.DataFrame({'Item': list(frequency.keys()), 'Frequency': list(frequency.values())})
    df = df.sort_values('Frequency', ascending=False)
    chart = alt.Chart(df).mark_bar().encode(
        x='Frequency',
        y=alt.Y('Item', sort='-x')
    ).properties(
        width=600,
        height=400,
        title='Item Frequency'
    )


    #st.title('Item Frequency')
    st.altair_chart(chart, use_container_width=True)
    st.write('### :orange[Tabel Of The Frequent Items] :')

    st.subheader('Frequency Table')
    st.write(df)



    st.write('### :orange[Top 10 Frequent Items] :')
    #graph 2
    sorted_data = [x for _, x in sorted(zip(support1, data), reverse=True)]
    sorted_support = sorted(support1, reverse=True)
    top_10_data = sorted_data[:10]
    top_10_support = sorted_support[:10]
    percentages = [support * 100 for support in top_10_support]
    colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#800000', '#008000', '#000080',
                  '#FFC0CB']
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, labels, _ = ax.pie(percentages, colors=colors, autopct='%1.1f%%', startangle=90, counterclock=False)
    ax.axis('equal')
    for wedge, label in zip(wedges, labels):
        wedge.set_edgecolor('white')
        label.set_fontweight('bold')
    ax.set_title('Top 10 Data Points with Support')
    legend_labels = [f'{sorted_data[i]}' for i in range(10)]
    ax.legend(wedges, legend_labels, title='Data Points', loc='center left', bbox_to_anchor=(1, 0.5))
    st.pyplot(fig)

    # fpgrowth

    data2 = []
    support2 = []

    fp_itemset = fpgrowth(mydf, min_support=0.02, use_colnames=True)

    fp_itemset["itemsets"] = fp_itemset["itemsets"].apply(lambda x: list(x))
    data2 = []
    support2 = []

    for i in range(fp_itemset.shape[0]):
        if len(fp_itemset.iloc[i]["itemsets"]) >= 2:
            data2.append(fp_itemset.iloc[i]["itemsets"])
            support2.append(fp_itemset.iloc[i]['support'])


    st.write('# In This Part we will Analyze the Data using :orange[FPgrowth Algorithm] :')
    # graph 1
    st.title('The FPgrowth Algorithm')
    st.write('### :orange[Item Frequency] :')

    flat_data = [item for sublist in data2 for item in sublist]
    frequency = dict(Counter(flat_data))
    df = pd.DataFrame({'Item': list(frequency.keys()), 'Frequency': list(frequency.values())})
    df = df.sort_values('Frequency', ascending=False)
    chart = alt.Chart(df).mark_bar().encode(
        x='Frequency',
        y=alt.Y('Item', sort='-x')
    ).properties(
        width=600,
        height=400,
        title='Item Frequency'
    )

    # st.title('Item Frequency')
    st.altair_chart(chart, use_container_width=True)
    st.write('### :orange[Tabel Of The Frequent Items] :')

    st.subheader('Frequency Table')
    st.write(df)

    st.write('### :orange[Top 10 Frequent Items] :')

    # graph 2
    sorted_data = [x for _, x in sorted(zip(support2, data2), reverse=True)]
    sorted_support = sorted(support2, reverse=True)
    top_10_data = sorted_data[:10]
    top_10_support = sorted_support[:10]
    percentages = [support * 100 for support in top_10_support]
    colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#800000', '#008000', '#000080',
              '#FFC0CB']
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, labels, _ = ax.pie(percentages, colors=colors, autopct='%1.1f%%', startangle=90, counterclock=False)
    ax.axis('equal')
    for wedge, label in zip(wedges, labels):
        wedge.set_edgecolor('white')
    label.set_fontweight('bold')
    ax.set_title('Top 10 Data Points with Support')
    legend_labels = [f'{sorted_data[i]}' for i in range(10)]
    ax.legend(wedges, legend_labels, title='Data Points', loc='center left', bbox_to_anchor=(1, 0.5))
    st.pyplot(fig)

#----------------------------------------------------------------------------------------------------

    #hmine
    data3 = []
    support3 = []


    hmine_itemset = hmine(mydf, min_support=0.02, use_colnames=True)

    hmine_itemset["itemsets"] = hmine_itemset["itemsets"].apply(lambda x: list(x))


    for i in range(hmine_itemset.shape[0]):
        if len(hmine_itemset.iloc[i]["itemsets"]) >= 2:
            data3.append(hmine_itemset.iloc[i]["itemsets"])
            support3.append(hmine_itemset.iloc[i]['support'])

    st.write('# In This Page we will Analyze the Data using :orange[Hmine Algorithm] :')
    #graph 1
    st.title('The Hmine Algorithm')
    st.write('### :orange[Item Frequency] :')

    flat_data = [item for sublist in data3 for item in sublist]
    frequency = dict(Counter(flat_data))
    df = pd.DataFrame({'Item': list(frequency.keys()), 'Frequency': list(frequency.values())})
    df = df.sort_values('Frequency', ascending=False)
    chart = alt.Chart(df).mark_bar().encode(
        x='Frequency',
        y=alt.Y('Item', sort='-x')
    ).properties(
        width=600,
        height=400,
        title='Item Frequency'
    )


    #st.title('Item Frequency')
    st.altair_chart(chart, use_container_width=True)
    st.write('### :orange[Tabel Of The Frequent Items] :')

    st.subheader('Frequency Table')
    st.write(df)

    st.write('### :orange[Top 10 Frequent Items] :')
    #graph 2
    sorted_data = [x for _, x in sorted(zip(support3, data3), reverse=True)]
    sorted_support = sorted(support3, reverse=True)
    top_10_data = sorted_data[:10]
    top_10_support = sorted_support[:10]
    percentages = [support * 100 for support in top_10_support]
    colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#800000', '#008000', '#000080',
                  '#FFC0CB']
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, labels, _ = ax.pie(percentages, colors=colors, autopct='%1.1f%%', startangle=90, counterclock=False)
    ax.axis('equal')
    for wedge, label in zip(wedges, labels):
        wedge.set_edgecolor('white')
        label.set_fontweight('bold')
    ax.set_title('Top 10 Data Points with Support')
    legend_labels = [f'{sorted_data[i]}' for i in range(10)]
    ax.legend(wedges, legend_labels, title='Data Points', loc='center left', bbox_to_anchor=(1, 0.5))
    st.pyplot(fig)




