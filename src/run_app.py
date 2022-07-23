import pandas as pd
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
#from sklearn.neighbors import NearestNeighbors
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras.backend as K
import boto3
import pickle
from google.cloud import storage
import Markowitz as mk


#aws_access = os.getenv("aws_access_key_id")   
#aws_secret = os.getenv("aws_secret_access_key")



#estrategia media movil
def movaverage_states(vseries, win):
    sma = vseries.rolling(win).mean()
    signal_states = (vseries > sma).astype(float)
    trading_states = signal_states.shift(1)
    trading_states.iloc[0] = 0
    return trading_states

def state_return_serie(price, states):
    """ 
"""
    real_serie = price.dropna()
    ret = real_serie.pct_change()
    ret.iloc[0] = 0
    ret_series = ret * states
    return ret_series

def IntToString(l):
        return list(map(str, l))
    
def cMSE(r, r_hat):
    obs = tf.equal(r,0)
    clr = tf.math.logical_not(obs)
    r = tf.boolean_mask(r, clr)
    r_hat = tf.boolean_mask(r_hat, clr)

    value_diff = r - r_hat

    cMse = K.mean(value_diff * value_diff)

    return cMse




#s3client = boto3.client('s3', 
#                        aws_access_key_id = aws_access, 
#                        aws_secret_access_key = aws_secret
#                       )

#response = s3client.get_object(Bucket='tfmmiax', Key='tablero.pkl')
#body = response['Body'].read()
#tablero = pickle.loads(body)

#response = s3client.get_object(Bucket='tfmmiax', Key='precios.pkl')
#body = response['Body'].read()
#precios = pickle.loads(body)

#response = s3client.get_object(Bucket='tfmmiax', Key='tabla_usuarios_fondos.pkl')
#body = response['Body'].read()
#tabla_completa = pickle.loads(body)

#response = s3client.get_object(Bucket='tfmmiax', Key='msci.pkl')
#body = response['Body'].read()
#msci = pickle.loads(body)




##descargamos desde google storage tablas de datos y modelos de ML y DL

storage_client = storage.Client.from_service_account_json(json_credentials_path='./tfm-roboadvisor-2c0a25f2dd58.json')
bucket = storage_client.bucket('tfm-miax7')

fichero = 'tabla_usuarios_fondos.pkl'
blob = bucket.blob(fichero)
blob.download_to_filename(fichero)
tabla_completa = pd.read_pickle(fichero)

fichero = 'tablero.pkl'
blob = bucket.blob(fichero)
blob.download_to_filename(fichero)
tablero = pd.read_pickle(fichero)

fichero = 'precios.pkl'
blob = bucket.blob(fichero)
blob.download_to_filename(fichero)
precios = pd.read_pickle(fichero)

fichero = 'msci.pkl'
blob = bucket.blob(fichero)
blob.download_to_filename(fichero)
msci = pd.read_pickle(fichero)


fichero = 'usuarios_cercanos.pkl'
blob = bucket.blob(fichero)
blob.download_to_filename(fichero)
loaded_model = pd.read_pickle(fichero)


fichero = 'my_autoencoder_1500.h5'
blob = bucket.blob(fichero)
blob.download_to_filename(fichero)
loaded_model_autoencoder = tf.keras.models.load_model(fichero, custom_objects={'cMSE': cMSE})

#inicializamos los dataframes que necesitaremos
usuarios = pd.DataFrame(columns=['perfil', 'preferencia_pais', 'preferencia_subcategory','Vola','Beta','calmar_ratio','Tracking_Error','Information_ratio', 'sortino_ratio', 'maxDrawDown_ratio', 'Omega'])
usuarios.loc[len(usuarios.index)] = 0,0,0,0,0,0,0,0,0,0,0
cartera = tabla_completa.iloc[0,13:]
cartera[:] = 0
fondos_elegir = pd.DataFrame(columns=['id', 'nombre'])
cartera_elegir = pd.DataFrame(columns=['id', 'nombre'])

external_stylesheets = ['tema_css.css']


def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


app.layout = html.Div(children=[
    html.H1(
        id='cabecera',
        children='MIAX 7  -  TFM AntBee Roboadvisor',
    ),    
    html.H6(
        children='Ana Isabel Torres Trujillo',
    ),
    
    html.Div(children = [
        html.H4("Perfíl:"),
        dcc.Dropdown(
            id='Perfil',
            options=[
                {'label': 'Conservador', 'value': 1.0},
                {'label': 'Moderado', 'value': 2.0},
                {'label': 'Agresivo', 'value': 3.0}
            ],
            value='Conservador'
        ),
               
        html.Br(),
        html.H4("País:"),
        dcc.Dropdown(
            id='Pais',
            options=[
                {'label': 'BRASIL', 'value': 92}, 
                {'label': 'CANADA', 'value': 1}, 
                {'label': 'CARIBE', 'value': 2}, 
                {'label': 'CHILE', 'value': 3}, 
                {'label': 'LATINOAMERICA MISC', 'value': 4}, 
                {'label': 'LATINOAMERICA', 'value': 5}, 
                {'label': 'MEXICO', 'value': 6}, 
                {'label': 'PANAMERICA', 'value': 7}, 
                {'label': 'USA', 'value': 8},
                {'label': 'ASIAN', 'value': 9}, 
                {'label': 'ASIA PASIFICO', 'value': 10},
                {'label': 'AUSTRALASIA', 'value': 11}, 
                {'label': 'AUSTRALIA', 'value': 12}, 
                {'label': 'ASIA EX JAPON', 'value':13}, 
                {'label': 'BAHREIN', 'value':14},
                {'label': 'CHINA', 'value':15}, 
                {'label': 'EMIRATOS ARABES UNIDOS', 'value':16},
                {'label': 'EGIPTO', 'value':17},
                {'label': 'FAR EAST', 'value':18},
                {'label': 'FAST EAST EX-JAPON', 'value':19}, 
                {'label': 'GCC', 'value':20}, 
                {'label': 'GREATER CHINA', 'value':21}, 
                {'label': 'HONG KONG', 'value':22}, 
                {'label': 'ISRAEL', 'value':23}, 
                {'label': 'INDIA', 'value':24}, 
                {'label': 'INDONESIA', 'value':25}, 
                {'label': 'JORDANIA', 'value':26}, 
                {'label': 'JAPON', 'value':27}, 
                {'label': 'KOREA', 'value':28}, 
                {'label': 'KUWAIT', 'value':29}, 
                {'label': 'LIBANO', 'value':30}, 
                {'label': 'ORIENTE MEDIO', 'value':31}, 
                {'label': 'MENA', 'value':32}, 
                {'label': 'MARRUECOS', 'value':33}, 
                {'label': 'MALASIA', 'value':34}, 
                {'label': 'OMAN', 'value':35}, 
                {'label': 'FILIPINAS', 'value':36}, 
                {'label': 'ARABIA SAUDITA', 'value':37}, 
                {'label': 'SINGAPUR', 'value':38}, 
                {'label': 'SINGAPUR Y MALASIA', 'value':39}, 
                {'label': 'THAILANDIA', 'value':40}, 
                {'label': 'TUNEZ', 'value':41}, 
                {'label': 'TAIWAN', 'value':42}, 
                {'label': 'VIETNAM', 'value':43}, 
                {'label': 'AUSTRIA', 'value':44}, 
                {'label': 'PAISES BALTICOS', 'value':45}, 
                {'label': 'BELGICA', 'value':46}, 
                {'label': 'BENELUX', 'value':47}, 
                {'label': 'SUIZA', 'value':48}, 
                {'label': 'REPUBLICA CHECA', 'value':49}, 
                {'label': 'ALEMANIA', 'value':50}, 
                {'label': 'DINAMARCA', 'value':51}, 
                {'label': 'EUROPA EMERGENTE', 'value':52}, 
                {'label': 'EUROPA EXSUIZA', 'value':53}, 
                {'label': 'EUROPA MISC', 'value':54}, 
                {'label': 'ESPAÑA', 'value':55}, 
                {'label': 'EUROPA', 'value':56}, 
                {'label': 'EUROPA EX-UK', 'value':57}, 
                {'label': 'ZONA EURO', 'value':58}, 
                {'label': 'FINLANDIA', 'value':59}, 
                {'label': 'FRANCIA', 'value':60}, 
                {'label': 'GRECIA', 'value':61}, 
                {'label': 'HUNGRIA', 'value':62},
                {'label': 'IBERICA', 'value':63}, 
                {'label': 'IRLANDA', 'value':64}, 
                {'label': 'ISLANDIA', 'value':65}, 
                {'label': 'ITALIA', 'value':66}, 
                {'label': 'LIETCHTENSTEIN', 'value':67}, 
                {'label': 'LUXEMBURGO', 'value':68}, 
                {'label': 'LUXEMBURGO', 'value':69}, 
                {'label': 'MALTA', 'value':70}, 
                {'label': 'PAISES BAJOS', 'value':71}, 
                {'label': 'NORUEGA', 'value':72}, 
                {'label': 'PAISES NORDICOS', 'value':73}, 
                {'label': 'POLONIA', 'value':74}, 
                {'label': 'PORTUGAL', 'value':75}, 
                {'label': 'RUSIA', 'value':76}, 
                {'label': 'SUECIA', 'value':77}, 
                {'label': 'ESCANDINAVIA', 'value':78}, 
                {'label': 'ESLOVAQUIA', 'value':79}, 
                {'label': 'TURQUIA', 'value':80}, 
                {'label': 'REINO UNIDO', 'value':81}, 
                {'label': 'REINO UNIDO', 'value':82}, 
                {'label': 'AFRICA', 'value':83}, 
                {'label': 'BRIC', 'value':84}, 
                {'label': 'GLOBAL EX AUSTRALIA', 'value':85}, 
                {'label': 'GLOBAL EMERGENTE', 'value':86}, 
                {'label': 'GLOBAL EX-US', 'value':87}, 
                {'label': 'GLOBAL', 'value':88}, 
                {'label': 'NUEVA ZELANDA', 'value':89}, 
                {'label': 'SUDAFRICA', 'value':90}, 
                {'label': 'EMEA', 'value':91}  
            ],
        ),
        html.Br(),
        html.H4("Categoría:"),
        dcc.Dropdown(
            id='Categoria',
            options=[
                {'label': 'alternativos_inmobiliario', 'value': 15}, 
                {'label': 'alternativos_liquidos', 'value':1},
                {'label':'mixto_flexible', 'value': 2}, 
                {'label':'mixtos_agresivos', 'value': 3}, 
                {'label':'mixtos_conservador', 'value': 4}, 
                {'label':'mixtos_equilibrado', 'value': 5}, 
                {'label':'monetario', 'value': 6}, 
                {'label':'otros', 'value': 7}, 
                {'label':'renta_fija_convertibles', 'value': 8}, 
                {'label':'renta_fija_corto_y_medio_plazo', 'value': 9}, 
                {'label':'renta_fija_emergente', 'value': 10}, 
                {'label':'renta_fija_high_yield', 'value': 11}, 
                {'label':'renta_fija_largo_plazo', 'value': 12}, 
                {'label':'renta_variable_sectorial', 'value': 13}, 
                {'label':'ver_zona_geografica', 'value': 14}
            ],
        ),
                
        ]),html.Br(),
    html.H3('Selecciona la importancia que le das a los diferentes ratios. SIN REPETICIÓN'),
    html.Div(children = [
        html.Div(children = [html.Br(),
            html.H4('Volatilidad'),
            dcc.Slider(
                min=1,
                max=9,
                id='Vola',
                marks={
                    1: '1',
                    2: '2',
                    3: '3',
                    4: '4',
                    5: '5',
                    6: '6',
                    7: '7',
                    8: '8',
                    9: '9'
                },
                
            ),
            html.H4('Beta'),
            dcc.Slider(
                min=1,
                max=9,
                id='Beta',
                marks={
                    1: '1',
                    2: '2',
                    3: '3',
                    4: '4',
                    5: '5',
                    6: '6',
                    7: '7',
                    8: '8',
                    9: '9'
                },
                
            ),
            html.H4('Calmar'),
            dcc.Slider(
                min=1,
                max=9,
                id='Calmar',
                marks={
                    1: '1',
                    2: '2',
                    3: '3',
                    4: '4',
                    5: '5',
                    6: '6',
                    7: '7',
                    8: '8',
                    9: '9'
                },
                
            ),
            html.H4('TrackingError'),
            dcc.Slider(
                min=1,
                max=9,
                id='TE',
                marks={
                    1: '1',
                    2: '2',
                    3: '3',
                    4: '4',
                    5: '5',
                    6: '6',
                    7: '7',
                    8: '8',
                    9: '9'
                },
                
            ),
        ]),
        html.Div(children = [html.Br(),
            html.H4('InformationRatio'),
            dcc.Slider(
                min=1,
                max=9,
                id='IR',
                marks={
                    1: '1',
                    2: '2',
                    3: '3',
                    4: '4',
                    5: '5',
                    6: '6',
                    7: '7',
                    8: '8',
                    9: '9'
                },
                
            ),
            html.H4('Sortino'),
            dcc.Slider(
                min=1,
                max=9,
                id='Sortino',
                marks={
                    1: '1',
                    2: '2',
                    3: '3',
                    4: '4',
                    5: '5',
                    6: '6',
                    7: '7',
                    8: '8',
                    9: '9'
                },
                
            ),
            html.H4('MaxDrawDown'),
            dcc.Slider(
                min=1,
                max=9,
                id='MaxDD',
                marks={
                    1: '1',
                    2: '2',
                    3: '3',
                    4: '4',
                    5: '5',
                    6: '6',
                    7: '7',
                    8: '8',
                    9: '9'
                },
                
            ),
            html.H4('Omega'),
            dcc.Slider(
                min=1,
                max=9,
                id='Omega',
                marks={
                    1: '1',
                    2: '2',
                    3: '3',
                    4: '4',
                    5: '5',
                    6: '6',
                    7: '7',
                    8: '8',
                    9: '9'
                },
                
            ),
        ]),
    ], style={'columnCount': 2}),
    html.Div(id='mensajes'),
    html.Br(),
    html.Button('Calcula fondos posibles', id='enviar', n_clicks=0),
    html.Div(id='container-button'),
    html.Br(),
    html.Div(children = [
        html.H4("Fondos a elegir de perfiles similares al tuyo:"),
        dcc.Dropdown(fondos_elegir.nombre.unique(),
            id="fondos_posibles", multi=True) ,
        html.Br(),
        html.Div(id='dd-output-container'),
        html.Br(),
        html.H4("Cartera propuesta según tus elecciones:"),
        dcc.Dropdown(cartera_elegir.nombre.unique(),
            id="cartera_sugerida",multi=True) ,
        html.Br(),
        html.Button('Calcula pesos', id='cartera', n_clicks=0),
        html.Br(),
        html.Div(id='result'),
        html.Br(),
        html.Br(),
        html.Div(children = [
        dcc.Graph(
            id='gr_fondos',        
        ),
        dcc.Graph(
            id='gr_rendimiento',        
        ),
        dcc.Graph(
            id='gr_comparativo',        
        ),
        ]), #, style={'columnCount': 3}
        html.Br(),
        html.Br(),
        
        ]               
        ), 
        
])

@app.callback( #al pulsar el boton generar cartera calcula markowitz y calcula el backtesting para pintar graficas
    Output('gr_comparativo', 'figure'),
    Output('gr_rendimiento', 'figure'),
    Output('gr_fondos', 'figure'),
    Input('cartera', 'n_clicks'),
    Input('fondos_posibles', 'value'),
    Input('cartera_sugerida', 'value')
)
def calcula_cartera(n_clicks, posibles, sugerida):
    if n_clicks>0:
        if(posibles is not None): 
            new_df1 = tablero[tablero['name'].isin(posibles)]
            if (sugerida is not None):
                new_df2 = tablero[tablero['name'].isin(sugerida)]
                new_df = pd.concat([new_df1, new_df2], ignore_index=True)
                dades = precios.loc[:,new_df.allfunds_id]
                rango_fechas = pd.to_datetime(precios['Unnamed: 0'], format='%d/%m/%Y')

                allocations, cartera, retorno_optimo, vola_optima, sharpe_optimo = mk.markowitz(dades.columns, dades, 10000)
                dades = precios.loc[:,cartera] #nos quedamos solo con los fondos que tienen allocation
                dades['Date'] = rango_fechas
                dades.set_index('Date')
                stock_df=dades
                stock_df.set_index('Date')
                
                    
                nombres = IntToString(stock_df.columns)
                stock_df.columns = nombres
                stock_df = stock_df.set_index('Date')
                
                msci.set_index('Date')
                # Indexamos las fechas
                msci.set_index(rango_fechas, inplace = True) 
                benchmark = msci['MSCI']
                
                
                stock_states = stock_df.apply(movaverage_states, win=50)
                
                sma_returns = [state_return_serie(stock_df[tk], stock_states[tk]) for tk in stock_df.columns]
                sma_returns_df = pd.concat(sma_returns, axis=1)
                sma_performance = (sma_returns_df + 1).cumprod()
                
                for i in range(sma_performance.shape[1]):
                    columna = sma_performance.columns[i]
                    sma_performance[columna]=sma_performance[columna] / allocations[allocations['tck']==np.double(columna)]['alloc'].values
                    
                figure1=px.line(sma_performance[sma_performance.columns], title='Fondos')
                
                                                
                #plot de los fondos en mismo grafico en funcion de su alloc
                
                porfolio_performance = sma_performance.sum(axis=1)
                figure2=px.line(porfolio_performance, title='Suma de rendimientos')
                
                #grafico de la suma del rendimiento
                
                relative_bm = benchmark/benchmark.iloc[0]
                
                estrategias = pd.DataFrame({
                    'SMA': porfolio_performance,
                    'MSCI': relative_bm
                })
                figure3=px.line(estrategias, title='Comparativo con MSCI')
                
                #plot comparativo benchmark
                
        
    return figure1, figure2, figure3 #'Retorno óptimo:', retorno_optimo, ' Volatilidad óptima: ', vola_optima, ' Ratio Sharpe óptimo: ', sharpe_optimo, figure2


@app.callback( #al elegir los fondos con el autoencoder calculamos los fondos para sugerir
    Output('cartera_sugerida', 'options'),
    Input('fondos_posibles', 'value')
)
def update_eleccion(values):
    if(values is not None): 
        new_df = tablero[tablero['name'].isin(values)]
        #tendriamos ahroa que ver que fondos hay para marcar la columna a la que pertenecen  
        cartera = tabla_completa.iloc[0,13:]
        cartera[:] = 0
        
        for row in new_df.itertuples():
            cartera[str(int(row.allfunds_id))] = 1

        cartera = pd.DataFrame(cartera)
        cartera = cartera.T
        
        r_hat = loaded_model_autoencoder.predict(cartera) * (cartera == 0)
        #r_hat = pd.DataFrame(r_hat)
        #r_hat.columns = cartera.columns
        r_hat2 = r_hat.T
        r_hat2 = r_hat2.sort_values(by=[0], ascending = False)
        r_hat2[0][10:] = 0.0
        r_hat2[0][0:10] = 1.0
        r_hat = r_hat2.T
        
        
        #lo pasamos por nuestro modelo para que nos sugiera cartera
        fondos_new = []
        for column in r_hat:
            if(r_hat[column][0]==1.0):
                fondos_new.append(column)
            
        for fondo in fondos_new:       
            reg = pd.DataFrame(tablero[tablero['allfunds_id'] == np.double(fondo)])
            df = reg.loc[:,['allfunds_id','name']] 
            if len(df) > 0:     
                cartera_elegir.loc[len(cartera_elegir.index)] = int(df.iloc[0,0]), df.iloc[0,1]
        
    return cartera_elegir.nombre.unique()


@app.callback( #al pulsar el boton mediante el modelo k-vecinos nos calcula los fondos que podriamos añadir segun similitud
    Output('fondos_posibles', 'options'),
    Input('enviar', 'n_clicks'),
    State('Omega', 'value'),
    State('MaxDD', 'value'),
    State('Sortino', 'value'),
    State('IR', 'value'),
    State('TE', 'value'),
    State('Calmar', 'value'),
    State('Beta', 'value'),
    State('Vola', 'value'),
    State('Perfil', 'value'),
    State('Pais', 'value'),
    State('Categoria', 'value')    
)
def update_output_button(n_clicks, Omega, MaxDD, Sortino, IR, TE, Calmar, Beta, Vola,Perfil,Pais,Categoria):
    for column in usuarios:
        usuarios[column]=usuarios[column].astype(float)

    distancia, indice = loaded_model.kneighbors(usuarios)

    df = pd.DataFrame(columns=tabla_completa.columns)
    for i in range(1, 4):            
        df.loc[len(df.index)]=((tabla_completa.iloc[indice[0][i], :]))
    df = df.iloc[:,13:] #nos quedamos solo con los fondos descartamos la informacion del usuario
    fondos = []

    for column in df:
        if df[column][0]==1:
            fondos.append(column)
        if df[column][1]==1:
            fondos.append(column)
        if df[column][2]==1:
            fondos.append(column)
        
    
    
    for fondo in fondos:       
        reg = pd.DataFrame(tablero[tablero['allfunds_id'] == np.double(fondo)])
        df = reg.loc[:,['allfunds_id','name']]      
        fondos_elegir.loc[len(fondos_elegir.index)] = int(df.iloc[0,0]), df.iloc[0,1]
    
    return fondos_elegir.nombre.unique()



@app.callback( #con la información dada generamos un dataframe con el perfil de inversor
    Output('mensajes', 'children'),
    Input('Omega', 'value'),
    Input('MaxDD', 'value'),
    Input('Sortino', 'value'),
    Input('IR', 'value'),
    Input('TE', 'value'),
    Input('Calmar', 'value'),
    Input('Beta', 'value'),
    Input('Vola', 'value'),
    Input('Perfil', 'value'),
    Input('Pais', 'value'),
    Input('Categoria', 'value')
    )
def update_output(Omega, MaxDD, Sortino, IR, TE, Calmar, Beta, Vola,Perfil,Pais,Categoria):    
    if Omega != None:
        usuarios.iloc[0,10] = round(Omega) 
    if MaxDD != None:
        usuarios.iloc[0,9] = round(MaxDD)
    if Sortino != None:
        usuarios.iloc[0,8] = round(Sortino)
    if IR != None:
        usuarios.iloc[0,7] = round(IR)
    if TE != None:
        usuarios.iloc[0,6] = round(TE)
    if Calmar != None:
        usuarios.iloc[0,5] = round(Calmar)
    if Beta != None:
        usuarios.iloc[0,4] = round(Beta)
    if Vola != None:
        usuarios.iloc[0,3] = round(Vola)
    if Perfil != None:
        if Perfil == 'Conservador':
            usuarios.iloc[0,0] = 1.0
        elif Perfil == 'Moderado':
            usuarios.iloc[0,0] = 2.0
        elif Perfil == 'Agresivo':
            usuarios.iloc[0,0] = 3.0
        else:
            usuarios.iloc[0,0] = Perfil
    if Pais != None:
        usuarios.iloc[0,1] = Pais
    if Categoria != None:
        usuarios.iloc[0,2] = Categoria
    return 'Perfil:', Perfil, ' Pais: ', Pais, ' Categoria: ', Categoria, ' Vola: ', Vola



if __name__ == '__main__':
    app.run_server(host="0.0.0.0", debug=False, port=8080)
