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

# Criação do app Streamlit
st.set_page_config(page_title="Análise de gastos bancários", page_icon=":bank:", layout="wide")
st.title("Análise de gastos bancários")

# Sidebar
st.sidebar.title("Opções")

# Opção para exibir os dados do dataset original
if st.sidebar.checkbox("Exibir dados originais"):
    st.subheader("Dados originais")
    st.write(df)

# Opção para selecionar o número de clusters
n = st.sidebar.slider("Selecione o número de clusters:", min_value=2, max_value=10, value=2, step=1)

# Utilizar o número de clusters selecionado para realizar a clusterização
kmeans = KMeans(n_clusters=n, random_state=42)
clusters = kmeans.fit_predict(final_df_norm)

# Adicionar a coluna de clusters no dataframe original
df['Cluster'] = kmeans.labels_


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

# previsão de gastos de um novo cliente:

# Receber os dados do novo cliente através de um formulário
if st.sidebar.checkbox("Gerar nova previsão de gastos"):
    st.write("Preencha as informações do novo cliente:")
    customer_age = st.number_input("Idade", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Gênero", options=['M', 'F'])
    dependent_count = st.number_input("Número de dependentes", min_value=0, max_value=20, value=0)
    education_level = st.selectbox("Nível de educação", options=['Graduate', 'High School', 'Unknown', 'Uneducated', 'College', 'Post-Graduate', 'Doctorate'])
    marital_status = st.selectbox("Estado civil", options=['Married', 'Single', 'Unknown', 'Divorced'])
    income_category = st.selectbox("Categoria de renda", options=['$60K - $80K', 'Less than $40K', '$80K - $120K', '$40K - $60K', '$120K +', 'Unknown'])
    card_category = st.selectbox("Categoria do cartão", options=['Blue', 'Gold', 'Silver', 'Platinum'])
    months_on_book = st.number_input("Meses como cliente", min_value=1, max_value=500, value=20)
    total_relationship_count = st.number_input("Total de produtos bancários contratados", min_value=1, max_value=20, value=1)
    months_inactive_12_mon = st.number_input("Meses de inatividade nos últimos 12 meses", min_value=0, max_value=12, value=1)
    contacts_count_12_mon = st.number_input("Número de contatos nos últimos 12 meses", min_value=0, max_value=100, value=0)
    credit_limit = st.number_input("Limite de crédito estimado:", min_value=0, max_value=100000, step=500)

    # Realiza a previsão de gastos do usuário com base nos dados informados
    user_data = pd.DataFrame({
        'Customer_Age': [customer_age],
        'Gender': [gender],
        'Dependent_count': [dependent_count],
        'Education_Level': [education_level],
        'Marital_Status': [marital_status],
        'Income_Category': [income_category],
        'Card_Category': [card_category],
        'Months_on_book': [months_on_book],
        'Total_Relationship_Count': [total_relationship_count],
        'Months_Inactive_12_mon': [months_inactive_12_mon],
        'Contacts_Count_12_mon': [contacts_count_12_mon],
        'Credit_Limit': [credit_limit],
        'Total_Revolving_Bal': [0],
        'Avg_Utilization_Ratio': [0]
    })
    
    user_data_encoded = encoder.transform(user_data[cat_cols])
    user_data_encoded_df = pd.DataFrame(user_data_encoded.toarray(), columns=encoder.get_feature_names_out(cat_cols))
    user_data_numeric_df = user_data[cluster_cols].reset_index(drop=True)
    user_data_final_df = pd.concat([user_data_numeric_df, user_data_encoded_df], axis=1)
    user_data_final_df_norm = scaler.transform(user_data_final_df)

    user_cluster = kmeans.predict(user_data_final_df_norm)

    user_cluster_data = df[df['Cluster'] == user_cluster[0]]
    user_predicted_spending = user_cluster_data['Total_Trans_Amt'].mean()

    # Exibe o valor de previsão de gastos do usuário
    st.success(f"A previsão de gastos mensal do usuário é de ${user_predicted_spending:.2f}")

