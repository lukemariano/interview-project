from math import sqrt

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Carregando os dados
df = pd.read_csv("./dataset/BankChurners.csv")

# Selecionar somente as colunas que importam para a análise
cols_to_keep = ['Customer_Age', 'Gender', 'Dependent_count', 'Education_Level', 'Marital_Status', 'Income_Category',
                'Card_Category', 'Months_on_book', 'Total_Relationship_Count', 'Months_Inactive_12_mon',
                'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal', 'Avg_Utilization_Ratio', 
                'Attrition_Flag', 'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Open_To_Buy']


df = df[cols_to_keep]

# Selecionar as colunas categóricas para aplicar o One-Hot Encoding
cat_cols = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']

# Aplicar o One-Hot Encoding
encoder = OneHotEncoder(drop='first')
encoded_cols = encoder.fit_transform(df[cat_cols])

# Transformar o resultado em um dataframe pandas
encoded_df = pd.DataFrame(encoded_cols.toarray(), columns=encoder.get_feature_names_out(cat_cols))

# Selecionar as colunas necessárias para a clusterização
cluster_cols = [
    'Customer_Age', 
    'Dependent_count', 
    'Months_on_book', 
    'Total_Relationship_Count', 
    'Months_Inactive_12_mon', 
    'Contacts_Count_12_mon', 
    'Credit_Limit', 
    'Total_Revolving_Bal',                 
    'Avg_Utilization_Ratio']

# Concatenar os dados numéricos com as colunas categóricas transformadas pelo One-Hot Encoding
numeric_df = df[cluster_cols].reset_index(drop=True)
final_df = pd.concat([numeric_df, encoded_df], axis=1)

# Aplicar a normalização dos dados
scaler = StandardScaler()
final_df_norm = scaler.fit_transform(final_df)

n = 2

# Utilizar o número ótimo de clusters identificado para realizar a clusterização
kmeans = KMeans(n_clusters=n, random_state=42)
clusters = kmeans.fit_predict(final_df_norm)

# Adicionar a coluna de clusters no dataframe original
df['Cluster'] = kmeans.labels_

# Criação do app Streamlit
st.set_page_config(page_title="Análise de gastos bancários", page_icon=":bank:", layout="wide")

st.title("Análise de gastos bancários")

# Sidebar
st.sidebar.title("Opções")

# Opção para exibir os dados do dataset original
if st.sidebar.checkbox("Exibir dados originais"):
    st.subheader("Dados originais")
    st.write(df)

# Opção para exibir os dados clusterizados
if st.sidebar.checkbox("Exibir dados clusterizados"):
    st.subheader("Dados clusterizados")
    st.write(df.groupby('Cluster').mean())

if st.sidebar.checkbox("Exibir distribuição dos clusters"):
    st.subheader("Distribuição dos clusters")
    sns.histplot(df, x="Cluster", stat="probability", kde=True)
    plt.legend(title='Clusters', labels=[f'Cluster {i}' for i in range(n)])
    st.pyplot()

if st.sidebar.checkbox("Exibir previsão de gastos por usuário"):
    st.subheader("Previsão de gastos por usuário")

    # carregar o dataframe com a previsão de gastos por usuário
    pred_df = pd.read_csv("./dataset/predicted_spending.csv")

    # exibir a tabela com a previsão de gastos por usuário
    st.write(pred_df)

fig, ax = plt.subplots(figsize=(10, 8))

for i in range(n):
    ax.scatter(final_df_norm[clusters == i, 0], final_df_norm[clusters == i, 1], s=50, label=f'Cluster {i}')

ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, marker='*', label='Centroids')
ax.legend()
ax.set_title('Clusterização dos clientes de um banco')
ax.set_xlabel('Primeiro componente principal')
ax.set_ylabel('Segundo componente principal')

st.subheader("Clusterização dos clientes de um banco")
st.pyplot(fig)



