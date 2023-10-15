# %% 
# Importar as bibliotecas necessárias
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from feature_engine.encoding import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn import tree
import numpy as np


# Carregar os dados
df = pd.read_csv("datasets/bd_alunos_evadidos.csv", sep=";")

# %% 
# Explorar os dados
print("Informações sobre as colunas:")
print(df.dtypes)

# %%
print("\nValores ausentes:")
print(df.isnull().sum())

# %%
# Exploração das Variáveis Categóricas
plt.figure(figsize=(20, 6))

# Frequência da variável 'AREACURSO'
frequencia_areacurso = df['AREACURSO'].value_counts()
print("\nFrequência da variável 'AREACURSO':")
print(frequencia_areacurso)

# Gráfico de Barras
sns.barplot(x=frequencia_areacurso.index, y=frequencia_areacurso.values)
plt.title('Gráfico de Barras - Contagem por AREACURSO')
plt.xlabel('AREACURSO')
plt.ylabel('Frequência')

# %%
# Função para agrupar as categorias em "Exatas" ou "Humanas"
def agrupar_categorias(categoria):
    if categoria in ['Ciências Exatas e da Terra', 'Engenharias']:
        return 'Exatas'
    elif categoria in ['Ciências Sociais Aplicadas', 'Ciências Humanas']:
        return 'Humanas'
    else:
        return 'Outras'
    
# Criar uma nova coluna 'Grupo_area_curso'
df['Grupo_area_curso'] = df['AREACURSO'].apply(agrupar_categorias)
contagem_por_grupo_area_curso = df['Grupo_area_curso'].value_counts()
print("\nContagem por grupo 'Grupo_area_curso':")
print(contagem_por_grupo_area_curso)

# %%
# Frequência da variável 'ACAOAFIRMATIVA'
frequencia_areafirmativa = df['ACAOAFIRMATIVA'].value_counts()
print("\nFrequência da variável 'ACAOAFIRMATIVA':")
print(frequencia_areafirmativa)

# %%
# Gráfico de Barras
plt.figure(figsize=(20, 6))
sns.barplot(x=frequencia_areafirmativa.index, y=frequencia_areafirmativa.values)
plt.title('Gráfico de Barras - Contagem por ACAOAFIRMATIVA')
plt.xlabel('ACAOAFIRMATIVA')
plt.ylabel('Frequência')

# %%
# Função para agrupar as categorias em "Ampla Concorrência" ou "Ações Afirmativas"
def categorias_to_grupo(categoria):
    if categoria in ['AC', 'A0']:
        return 'Ampla Concorrência'
    else:
        return 'Ações Afirmativas'

# Criar uma nova coluna 'Grupo_criterio'
df['Grupo_criterio'] = df['ACAOAFIRMATIVA'].apply(categorias_to_grupo)
contagem_por_grupo_criterio = df['Grupo_criterio'].value_counts()
print("\nContagem por grupo 'Grupo_criterio':")
print(contagem_por_grupo_criterio)

# %%
# Exclusão de Colunas
colunas_excluir = ['CODALUNO', 'STATUSFORMACAO', 'CR', 'CURSO', 'CODTURNOINGRESSO', 'CODTURNOATUAL',
                   'DISCIPLINA', 'NOTADISC', 'RESULTDISC', 'PERIODODISC', 'ANODESVINCULACAO',
                   'SEMESTREDESVINCULACAO', 'BAIRRO', 'CEP', 'CIDADE', 'CHCURSADA', 'TRANCAMENTOS',
                   'TEMPOPERMANENCIA', 'NOME_CURSO', 'cep_destino', 'MOBILIDADE', 'Unnamed: 0',
                   'DISTANCIA_NUM', 'ACAOAFIRMATIVA', 'AREACURSO', 'TURNOATUAL']

df = df.drop(colunas_excluir, axis=1)

# %%
# Codificação One-Hot de Variáveis Categóricas
cat_features = ['Grupo_criterio', 'SEMESTREINGRESSO', 'COR', 'ESTADOCIVIL', 'SEXO', 'Grupo_area_curso']
df[cat_features] = df[cat_features].astype(str)
onehot = OneHotEncoder(variables=cat_features)
X_transform = onehot.fit_transform(df)
X_transform

# %%
# Normalização Min-Max das variáveis numéricas
min_max = MinMaxScaler()
min_max.set_output(transform='pandas')
X_transform = min_max.fit_transform(X_transform)
X_transform

# %%
model_cluster = KMeans(n_clusters=12, random_state=60, max_iter=1000)  # Defina a semente aleatória no modelo KMeans
visualizer = KElbowVisualizer(model_cluster, k=(2, 12), random_state=60)  # Defina a semente aleatória no KElbowVisualizer
visualizer.fit(X_transform)
visualizer.show()

# %%
# Criar e ajustar o modelo KMeans com o número ideal de clusters
model_cluster = KMeans(n_clusters=visualizer.elbow_value_, random_state=65)
model_cluster.fit(X_transform)
cluster_labels = model_cluster.predict(X_transform)

# %%
# Adicionar os rótulos dos clusters ao DataFrame
X_transform['cluster_name'] = cluster_labels
print (X_transform['cluster_name'])

# %%
# Estatísticas descritivas para cada cluster
summary = X_transform.groupby(['cluster_name']).mean()
print(summary)

# %%
# Criar um mapa de calor das estatísticas dos clusters
plt.figure(figsize=(10, 6))
sns.heatmap(summary, annot=True, cmap='viridis', fmt=".2f")
plt.xlabel('Estatísticas')
plt.ylabel('Clusters')
plt.title('Mapa de Calor das Estatísticas dos Clusters')
plt.show()

# %%
# Calcular o Silhouette Score para os clusters
silhouette_avg = silhouette_score(X_transform.drop('cluster_name', axis=1), cluster_labels)
print(f'Silhouette Score: {silhouette_avg}')

# %%
np.random.seed(60)
clf = tree.DecisionTreeClassifier(random_state=60) #algoritmo de arvore
clf.fit(X_transform[X_transform.columns.tolist()[:-1]], X_transform['cluster_name']) # fit arvore
features_importance = pd.Series(clf.feature_importances_, index=X_transform.columns.tolist()[:-1]) #pega a importancia das variaveis
features_importance = features_importance.sort_values(ascending=False) #ordena
print(features_importance)

# %%
# Substitua 'features_selecionadas' pelas features mais importantes que você deseja usar
# Defina um limite de importância
limite_importancia = 0.01

# Selecione as características com importância acima do limite
features_selecionadas = features_importance[features_importance >= limite_importancia].index.tolist()

# Crie um novo DataFrame com as features selecionadas
X_selected_features = X_transform[features_selecionadas]
X_selected_features

# %%

# model_cluster = KMeans(n_clusters=12, random_state=65, max_iter=1000)  # Defina a semente aleatória no modelo KMeans
#visualizer = KElbowVisualizer(model_cluster, k=(2, 12), random_state=65)  # Defina a semente aleatória no KElbowVisualizer
#visualizer.fit(X_selected_features)
#visualizer.show()

# %%
# Criar e ajustar o modelo KMeans com o número ideal de clusters
# model_cluster_selected = KMeans(n_clusters=visualizer.elbow_value_, random_state=65)
model_cluster.fit(X_selected_features)
cluster_labels_selected = model_cluster.predict(X_selected_features)

# Adicione os rótulos dos clusters ao DataFrame original ou a um novo DataFrame
X_selected_features['cluster_name_selected'] = cluster_labels_selected

# Visualize ou analise os resultados da clusterização
print(X_selected_features['cluster_name_selected'].value_counts())

# %%
# Estatísticas descritivas para cada cluster
summary_selected = X_selected_features.groupby(['cluster_name_selected']).mean()
print(summary_selected)

# %%
# Criar um mapa de calor das estatísticas dos clusters
plt.figure(figsize=(10, 6))
sns.heatmap(summary_selected, annot=True, cmap='viridis', fmt=".2f")
plt.xlabel('Estatísticas')
plt.ylabel('Clusters')
plt.title('Mapa de Calor das Estatísticas dos Clusters')
plt.show()

# %%
# Calcular o Silhouette Score para os clusters
silhouette_avg = silhouette_score(X_selected_features.drop('cluster_name_selected', axis=1), cluster_labels)
print(f'Silhouette Score: {silhouette_avg}')

# %%
X_selected_features
features = X_selected_features.columns.tolist()[:-1]
features

target = 'cluster_name_selected'

# %%

clf = tree.DecisionTreeClassifier()

clf.fit(X_selected_features[features], X_selected_features[target])

# %%

plt.figure(dpi=400)
tree.plot_tree(clf,feature_names=features,  
                filled=True)
# %%
