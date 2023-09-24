# %% 
# Importar as bibliotecas necessárias
import pandas as pd
from sklearn import preprocessing
from feature_engine.encoding import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from yellowbrick.cluster import KElbowVisualizer
import seaborn as sns
from sklearn import cluster
import matplotlib.pyplot as plt

# %%
df = pd.read_csv("C:/Users/Mariana Moledo/Documents/GitHub/tcc_mba_cd/datasets/bd_alunos_evadidos.csv",sep=';', encoding='utf-8')
df.columns

# %%
df.dtypes

# %%
# Verificar a presença de valores ausentes
print(df.isnull().sum())

# %% [markdown]
# Exploração das Variáveis Categóricas
# %%
#Frequência Estado Civil
frequencia_areacurso = df['AREACURSO'].value_counts()
print(frequencia_areacurso)

#Proporção Estado Civil
proporcao_areacurso = df['AREACURSO'].value_counts(normalize=True)
print(proporcao_areacurso)

#Gráfico de Barras
plt.figure(figsize=(20, 6))
sns.barplot(x=frequencia_areacurso.index, y=frequencia_areacurso.values)
plt.title('Gráfico de Barras - Contagem por AREACURSO') #Adicionando título ao gráfico
plt.xlabel('AREACURSO') #Adicionando rótulos aos eixos
plt.ylabel('Frequência') #Adicionando rótulos aos eixos

# %%
# Crie uma função para agrupar as categorias em "Exatas" ou "Humanas"
def agrupar_categorias(categoria):
    if categoria in ['Ciências Exatas e da Terra', 'Engenharias']:
        return 'Exatas'
    elif categoria in ['Ciências Sociais Aplicadas', 'Ciências Humanas']:
        return 'Humanas'
    else:
        return 'Outras'

# Aplique a função para criar uma nova coluna 'Grupo' no DataFrame
df['Grupo_area_curso'] = df['AREACURSO'].apply(agrupar_categorias)

# Visualize o DataFrame com a nova coluna 'Grupo'
print(df[['AREACURSO', 'Grupo_area_curso']])

# Use a função value_counts() para contar as ocorrências em cada grupo
contagem_por_grupo_area_curso = df['Grupo_area_curso'].value_counts()

# Exiba o resultado
print(contagem_por_grupo_area_curso)

# %%
df.head()

# %%
#Frequência Estado Civil
frequencia_areafirmativa = df['ACAOAFIRMATIVA'].value_counts()
print(frequencia_areafirmativa)

#Proporção Estado Civil
proporcao_areafirmativa = df['ACAOAFIRMATIVA'].value_counts(normalize=True)
print(proporcao_areafirmativa)

#Gráfico de Barras
plt.figure(figsize=(20, 6))
sns.barplot(x=frequencia_areafirmativa.index, y=frequencia_areafirmativa.values)
plt.title('Gráfico de Barras - Contagem por AREACURSO') #Adicionando título ao gráfico
plt.xlabel('AREACURSO') #Adicionando rótulos aos eixos
plt.ylabel('Frequência') #Adicionando rótulos aos eixos

# %% [markdown]
# Vamos criar dois grupos: "Ampla Concorrência" e "Ações Afirmativas". As categorias "Ampla Concorrência" incluiriam aquelas em que não há critérios específicos de renda ou etnia, enquanto as categorias "Ações Afirmativas" incluiriam aquelas que têm critérios específicos de renda, etnia ou deficiência. 

# %%
# Crie uma função para agrupar as categorias
def categorias_to_grupo(categoria):
    if categoria in ['AC', 'A0']:
        return 'Ampla Concorrência'
    else:
        return 'Ações Afirmativas'

# Aplique a função para criar uma nova coluna 'Grupo' no DataFrame
df['Grupo_criterio'] = df['ACAOAFIRMATIVA'].apply(categorias_to_grupo)

# Visualize o DataFrame com a nova coluna 'Grupo'
print(df[['ACAOAFIRMATIVA', 'Grupo_criterio']])

# Use a função value_counts() para contar as ocorrências em cada grupo
contagem_por_grupo_criterio = df['Grupo_criterio'].value_counts()

# Exiba o resultado
print(contagem_por_grupo_criterio)

# %%
df.head()

# %%
# Exclusão de Colunas
colunas_excluir = ['CODALUNO',
                   'STATUSFORMACAO',
                   'CR',
                   'CURSO',
                   'CODTURNOINGRESSO',
                   'CODTURNOATUAL',
                   'DISCIPLINA',
                   'NOTADISC',
                   'RESULTDISC',
                   'PERIODODISC',
                   'ANODESVINCULACAO',
                   'SEMESTREDESVINCULACAO',
                   'BAIRRO',
                   'CEP',
                   'CIDADE',
                   'CHCURSADA',
                   'TRANCAMENTOS',
                   'TEMPOPERMANENCIA',
                   'NOME_CURSO',
                   'cep_destino',
                   'MOBILIDADE',
                   'Unnamed: 0',
                   'DISTANCIA_NUM',
                   'ACAOAFIRMATIVA',
                   'AREACURSO',
                   'TURNOATUAL'
                   ]

df = df.drop(colunas_excluir, axis=1)

# %%
df.head()
# %% 
# Seleciona colunas do tipo 'object'
cat_features = ['Grupo_criterio',
                'SEMESTREINGRESSO',
                'COR',
                'ESTADOCIVIL',
                'SEXO',
                'Grupo_area_curso']
df[cat_features] = df[cat_features].astype(str)

#%%
# Codificação One-Hot de Variáveis Categóricas
onehot = OneHotEncoder(variables=cat_features)
X_transform = onehot.fit_transform(df)

# %%
X_transform.dtypes

# %%

# Normalização Min-Max das variáveis numéricas
min_max = preprocessing.MinMaxScaler()
min_max.set_output(transform='pandas')

X_transform = min_max.fit_transform(X_transform)

X_transform

# %%
# Seleção de Características por Variância para as variáveis categóricas
var_feature_importance = VarianceThreshold(0.21)
var_feature_importance.set_output(transform='pandas')
X_transform_filtered = var_feature_importance.fit_transform(
    X_transform)
X_transform_filtered

# %%

# criação de uma instância do modelo KMeans para realizar a clusterização.
model_cluster = cluster.KMeans(random_state=42, max_iter=1000,)
model_cluster

# %%
visualizer = KElbowVisualizer(model_cluster, k=(2, 12))
visualizer.fit(X_transform_filtered)
visualizer.show()

# %%
model_cluster = cluster.KMeans(n_clusters=visualizer.elbow_value_)

# %%
model_cluster.fit(X_transform_filtered)


# %%
cluster_labels = model_cluster.predict(X_transform_filtered)
cluster_labels


# %%
X_transform_filtered['cluster_name'] = model_cluster.labels_
X_transform_filtered

# Estatísticas descritivas para cada cluster
summary = X_transform_filtered.groupby(['cluster_name']).mean()

# %%
# Visualização das estatísticas em um mapa de calor
sns.heatmap(summary, cmap='viridis')

# %%
# Criar subconjuntos de dados para cada cluster
clusters_data = []
for cluster_id in range(5):  # Defina o número de clusters desejado
    cluster_data = X_transform_filtered[X_transform_filtered['cluster_name'] == cluster_id].drop(
        'cluster_name', axis=1)
    clusters_data.append(cluster_data)

# Aplicar PCA para cada subconjunto de dados do cluster
pca_models = []
for cluster_data in clusters_data:
    pca = PCA(n_components=2)
    pca.fit(cluster_data)
    pca_models.append(pca)

# Plotar os resultados do PCA para cada cluster
for cluster_id, pca_model in enumerate(pca_models):
    cluster_data = clusters_data[cluster_id]
    reduced_data = pca_model.transform(cluster_data)

    plt.scatter(reduced_data[:, 0], reduced_data[:, 1],
                label=f'Cluster {cluster_id}') 

plt.legend()
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('Visualização dos Clusters Após PCA')
plt.show()


 # %%
from sklearn.metrics import silhouette_score

# Calcule o Silhouette Score para os clusters gerados pelo modelo
silhouette_avg = silhouette_score(X_transform_filtered.drop('cluster_name', axis=1), model_cluster.labels_)

# Imprima o valor do Silhouette Score
print(f'Silhouette Score: {silhouette_avg}')

#A interpretação do Silhouette Score pode ser feita da seguinte forma:
# Se o Silhouette Score estiver próximo de +1, os clusters estão bem separados.
# Se o Silhouette Score estiver próximo de 0, os clusters têm sobreposição.
# Se o Silhouette Score estiver próximo de -1, os pontos foram atribuídos ao cluster errado.
# %%

