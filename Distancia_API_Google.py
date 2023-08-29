# %% 
import pandas as pd
import googlemaps
import time
# %%

df = pd.read_excel ('C:/Users/Mariana Moledo/Documents/Dataset-UFF-Graduacao-Evadidos-CEPs_Validos_2.xlsx')

#%%
#Para pegarmos os ceps das universidades para calcular a distancia usamos o link https://www.uff.br/?q=cursos-dados-completos&field_turno_value=All&field_titulacao_curso_value=All&field_tipo_de_curso_value=All&field_sigla_value=&field_c_digo_e_mec_value=

# %%
APIKey = 'AIzaSyCCMPqOaRTOJHnY-xxusl0FsPqxE15PSE4'

gmaps = googlemaps.Client(key=APIKey)

print(gmaps)

# %%

# Função para obter as coordenadas geográficas (latitude e longitude) de um CEP usando a API do Google Geocoding
def get_coordinates_from_cep(cep):
    geocode_result = gmaps.geocode(cep)
    if geocode_result:
        location = geocode_result[0]['geometry']['location']
        return location['lat'], location['lng']
    else:
        return None, None

# Adicionar colunas para latitude e longitude de origem e destino
df['lat_origem'] = None
df['lon_origem'] = None
df['lat_destino'] = None
df['lon_destino'] = None
df['distancia'] = None

# Número de registros a serem processados de uma vez (10 neste exemplo)
batch_size = 100

# Iterar pelas linhas do DataFrame em lotes de tamanho batch_size
for batch_start in range(0, len(df), batch_size):
    batch_end = min(batch_start + batch_size, len(df))
    batch_df = df.iloc[batch_start:batch_end]

    # Novo DataFrame para armazenar os resultados do lote atual
    result_df = batch_df.copy()

    # Iterar pelas linhas do lote e calcular a distância
    for index, row in batch_df.iterrows():
        cep_origem = row['cep_origem']
        cep_destino = row['cep_destino']

        # Obter as coordenadas de origem e destino
        lat_origem, lon_origem = get_coordinates_from_cep(cep_origem)
        lat_destino, lon_destino = get_coordinates_from_cep(cep_destino)

        # Armazenar as coordenadas no DataFrame do lote atual
        result_df.at[index, 'lat_origem'] = lat_origem if lat_origem is not None else -1
        result_df.at[index, 'lon_origem'] = lon_origem if lon_origem is not None else -1
        result_df.at[index, 'lat_destino'] = lat_destino
        result_df.at[index, 'lon_destino'] = lon_destino

        if lat_origem is not None and lon_origem is not None and lat_destino and lon_destino:
            # Calcular a distância usando a API distance_matrix do Google Maps
            result = gmaps.distance_matrix((lat_origem, lon_origem), (lat_destino, lon_destino))
        
            # Verificar se a API retornou um resultado válido
            if result['status'] == 'OK':
                try:
                    distance = result['rows'][0]['elements'][0]['distance']['text']
                    result_df.at[index, 'distancia'] = distance
                except KeyError:
                    # Tratamento de erro caso a chave 'distance' não exista no resultado
                    print(f"Erro: A chave 'distance' não existe na resposta da API para CEPs {cep_origem} e {cep_destino}.")
            else:
                # Tratamento de erro caso o resultado não seja OK (pode ser "OVER_QUERY_LIMIT" ou outro status de erro)
                print(f"Erro ao calcular distância para CEPs {cep_origem} e {cep_destino}. Status da API: {result['status']}")

        # Aguardar um intervalo para respeitar os limites da API
        time.sleep(0.5)  # Pausa de 0.5 segundos entre as requisições

    # Exibir o lote de resultados
    print(f"Processados registros {batch_start + 1} a {batch_end}")

    # Salvar o lote de resultados em um arquivo CSV
    result_df.to_csv(f'dados_batch_{batch_start}_{batch_end}.csv', index=False)
    result_df.to_excel(f'dados_batch_{batch_start}_{batch_end}.xlsx', index=False)

# Exibir o DataFrame completo com as informações de latitude, longitude e distância
print(df)

# %%
