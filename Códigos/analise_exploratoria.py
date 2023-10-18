# %%
import pandas as pd
from sklearn import feature_selection
from feature_engine.encoding import OneHotEncoder
from sklearn import preprocessing
from yellowbrick.cluster import KElbowVisualizer
import seaborn as sns
from sklearn import pipeline
from sklearn import cluster
import matplotlib.pyplot as plt

# %%
# Carregar os dados
file_path = "datasets/bd_alunos_evadidos.csv"
df = pd.read_csv(file_path, sep=';', encoding='utf-8')

# %%
df.dtypes

# %% 
df.columns 
# %%
# Exibir as primeiras linhas do DataFrame para entender a estrutura dos dados
print(df.head())

# %%
# Verificar tipos de dados e informações gerais
print(df.info())

# %%
# Estatísticas descritivas para variáveis numéricas
print(df.describe())

# %%
# Verificar a presença de valores ausentes
print(df.isnull().sum())

# %%
# Frequência da variável 'AREACURSO'
frequencia_areacurso = df['AREACURSO'].value_counts()

# Escolha uma paleta de cores com degradê
palette = sns.dark_palette("#69d", len(frequencia_areacurso))

# Crie o gráfico de barras horizontal
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=frequencia_areacurso.values, y=frequencia_areacurso.index, palette=palette, orient='h')

# Adicione rótulos aos dados
for i, v in enumerate(frequencia_areacurso.values):
    ax.text(v + 3, i, str(v), color='gray', va='center', fontsize=10)

plt.show()

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

# Gráfico de Barras Vertical
plt.figure(figsize=(8, 4))

# Crie o gráfico de barras vertical
sns.barplot(x=contagem_por_grupo_area_curso.values, y=contagem_por_grupo_area_curso.index, color='skyblue')

# Adicione rótulos aos dados
for i, v in enumerate(contagem_por_grupo_area_curso.values):
    plt.text(v + 3, i, str(v), color='black', va='center', fontsize=10)

# Mostrar o gráfico
plt.show()

# %%
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


# Gráfico de Barras Vertical
plt.figure(figsize=(8, 4))

# Crie o gráfico de barras vertical
sns.barplot(x=contagem_por_grupo_criterio.values, y=contagem_por_grupo_criterio.index, color='grey')

# Adicione rótulos aos dados
for i, v in enumerate(contagem_por_grupo_criterio.values):
    plt.text(v + 3, i, str(v), color='black', va='center', fontsize=10)

# Mostrar o gráfico
plt.show()

# %%
sexo_freq = df['SEXO'].value_counts()
sexo_percentual = (df['SEXO'].value_counts(normalize=True) * 100).round(2)

# Crie um DataFrame com as informações
tabela_frequencia_sexo = pd.DataFrame({'Frequência': sexo_freq, 'Percentual (%)': sexo_percentual})

# Exiba a tabela de frequência
print(tabela_frequencia_sexo)
#%%
# Tabela de frequência e percentual para a coluna "Estado Civil"
semestre_ingresso_freq = df['SEMESTREINGRESSO'].value_counts()
semestre_ingresso_freq_percentual = (df['SEMESTREINGRESSO'].value_counts(normalize=True) * 100).round(2)

# Crie um DataFrame com as informações
tabela_frequencia_semestre_ingresso = pd.DataFrame({'Frequência': semestre_ingresso_freq, 'Percentual (%)': semestre_ingresso_freq_percentual})

# Exiba a tabela de frequência
print(tabela_frequencia_semestre_ingresso)

#%%
# Tabela de frequência e percentual para a coluna "Idade"
idade_freq = df['IDADE'].value_counts()
idade_freq_percentual = (df['IDADE'].value_counts(normalize=True) * 100).round(2)

# Crie um DataFrame com as informações
tabela_frequencia_idade = pd.DataFrame({'Frequência': idade_freq, 'Percentual (%)': idade_freq_percentual})

# Exiba a tabela de frequência
print(tabela_frequencia_idade)

#%%
# Tabela de frequência e percentual para a coluna "ESTADOCIVIL"
ESTADOCIVIL_freq = df['ESTADOCIVIL'].value_counts()
ESTADOCIVIL_freq_percentual = (df['ESTADOCIVIL'].value_counts(normalize=True) * 100).round(2)

# Crie um DataFrame com as informações
tabela_frequencia_ESTADOCIVIL = pd.DataFrame({'Frequência': ESTADOCIVIL_freq, 'Percentual (%)': ESTADOCIVIL_freq_percentual})

# Exiba a tabela de frequência
print(tabela_frequencia_ESTADOCIVIL)

#%%
# Tabela de frequência e percentual para a coluna "TURNOATUAL"
TURNOATUAL_freq = df['TURNOATUAL'].value_counts()
TURNOATUAL_freq_percentual = (df['TURNOATUAL'].value_counts(normalize=True) * 100).round(2)

# Crie um DataFrame com as informações
tabela_frequencia_TURNOATUAL = pd.DataFrame({'Frequência': TURNOATUAL_freq, 'Percentual (%)': TURNOATUAL_freq_percentual})

# Exiba a tabela de frequência
print(tabela_frequencia_TURNOATUAL)
# %%
#plb.plot(dataEvadidos.groupby(['TEMPOPERMANENCIA']))
g= sns.catplot(x = "TEMPOPERMANENCIA", y = "CHCURSADA", data = df, kind = "bar")
g.set_ylabels("Carga Horária Cursada (Evadidos)")
g.set_xlabels("Tempo de Permanência (Evadidos)")
plt.show()
# %%
