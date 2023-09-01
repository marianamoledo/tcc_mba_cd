# %%
import pandas as pd
from sklearn import feature_selection
from feature_engine.encoding import OneHotEncoder
from sklearn import preprocessing

# Definindo opções de exibição para o Pandas
pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 1000)

df_fato = pd.read_excel("Dataset-UFF-Graduacao-Evadidos-CEPs_Validos.xlsx")
df_curso = pd.read_csv("Classificacao_cursos.csv",sep=';', encoding='latin-1')

# %%

colunas_excluir = ['CODALUNO',
                   'STATUSFORMACAO',
                   'CR',
                   'DISTANCIA',
                   'STATUS_CURSO',
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
                   'cep_origem',
                   'CIDADE',
                   'CHCURSADA',
                   'TRANCAMENTOS',
                   'TEMPOPERMANENCIA',
                   'NOME_CURSO',
                   'cep_destino',
                   'MOBILIDADE',
                   ]

# %%

df_join = df_fato.merge(df_curso, on='NOME_CURSO')
df_join = df_join.rename(columns={"Classificação":"AREACURSO"})

df_join = df_join.drop(colunas_excluir, axis=1)
# %%

cat_features = ['ACAOAFIRMATIVA',
                'TURNOATUAL',
                'ANOINGRESSO',
                'SEMESTREINGRESSO',
                'COR',
                'ESTADOCIVIL',
                'SEXO',
                'AREACURSO']

df_join[cat_features] = df_join[cat_features].astype(str)

onehot = OneHotEncoder(variables=cat_features)
X_transform = onehot.fit_transform(df_join)

min_max = preprocessing.MinMaxScaler()
min_max.set_output(transform='pandas')


X_transform = min_max.fit_transform(X_transform)

X_transform
# %%

var_feature_importance = feature_selection.VarianceThreshold(0.03)
var_feature_importance.set_output(transform='pandas')
X_transform_filter = var_feature_importance.fit_transform(X_transform)

X_transform_filter

# %%

X_transform_filter.var().reset_index().T
# %%
