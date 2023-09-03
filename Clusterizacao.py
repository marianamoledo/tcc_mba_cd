# %% Importação de Bibliotecas

import pandas as pd
from sklearn import feature_selection
from feature_engine.encoding import OneHotEncoder
from sklearn import preprocessing

# Definição de Opções de Exibição do Pandas:

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 1000)

# Carregamento de Dados
df = pd.read_csv("bd_alunos_evadidos.csv",sep=';', encoding='latin-1')
df.columns

# %% Exclusão de Colunas

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
                   'Unnamed: 0'
                   ]

# %%
df = df.drop(colunas_excluir, axis=1)

# %% Codificação One-Hot de Variáveis Categóricas
cat_features = ['ACAOAFIRMATIVA',
                'TURNOATUAL',
                'ANOINGRESSO',
                'SEMESTREINGRESSO',
                'COR',
                'ESTADOCIVIL',
                'SEXO',
                'AREACURSO']

df[cat_features] = df[cat_features].astype(str)

onehot = OneHotEncoder(variables=cat_features)
X_transform = onehot.fit_transform(df)

# Normalização Min-Max
# coloca todas as características numéricas no intervalo de 0 a 1
min_max = preprocessing.MinMaxScaler()
min_max.set_output(transform='pandas')


X_transform = min_max.fit_transform(X_transform)

X_transform
# %% Seleção de Características por Variância:

var_feature_importance = feature_selection.VarianceThreshold(0.015)
var_feature_importance.set_output(transform='pandas')
X_transform_filter = var_feature_importance.fit_transform(X_transform)
X_transform_filter

# %%
X_transform_filter.var().reset_index().T
# %%
