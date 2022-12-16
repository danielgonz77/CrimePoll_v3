import datetime
from distutils.log import debug
import math
from chardet import detect_all
from matplotlib import container
import numpy as np
import pandas as pd
import plotly.express as px
from scipy import stats
from datetime import date


import dash
import dash_core_components as dcc
#import dash_html_components as html
import dash_bootstrap_components as dbc
from dash import html
import dash_daq as daq

from dash.dependencies import Input, Output


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


#aire_map = pd.read_csv("../CSV_limpios/aire_v2.csv")
#crime_map = pd.read_csv("../CSV_limpios/crimen_v2.csv")

aire_map = pd.read_csv("./csv/aire_v2.csv")
crime_map = pd.read_csv("./csv/crimen_v2.csv")


df_all = pd.read_csv("./csv/df_air_crime_evaluation.csv")
#df_all = pd.read_csv("../CSV_limpios/df_air_crime_evaluation_o3.csv")

models = {
     "BENITO JUAREZ": [1, 214.9126391708, -0.0018374312, 0.0302102129, -0.1055739376, 0.0058998541, 24.6],
     "CUAJIMALPA DE MORELOS": [5, -40.077475915, -0.0020568653, -0.0054645416, 0.0204733694, 0.0007117387, 77.75],
     "CUAUHTEMOC": [8, -1429.8475079076, -0.0158129401, 0.029503595, 0.7116424221, 0.0032419824, 20.17],
     "IZTACALCO": [3, -449.6763203576, 0.0038919214, 0.0508705173, 0.2237442662, 0.0101261185, 27.55],
     "MIGUEL HIDALGO": [9, -1781.1780802088, 0.0027193564, 0.1020666467, 0.8840865336, 0.0021126176, 34.07],
     "TLAHUAC": [6, -267.7351537166, -0.0019352763, -0.0148895216, 0.1337588241, 0.000818561, 47.4],
     "TLALPAN": [4, -737.6545789514, 0.0059929009, 0.0060511696, 0.3673142828, -0.0034980614, 29.05],
     "VENUSTIANO CARRANZA": [2, -393.2098159588, -0.0052166789, -0.0281358348, 0.1964024774, 0.0030982005, 24.55],
     "AZCAPOTZALCO": [7, -159.1142355563, -0.0040754038, -0.0003299716, 0.0796972529, 0.0093918495, 27.84],
     "IZTAPALAPA": [10, 456.3869000885, 0.0044874625, -0.0774887946, -0.2225607627, 0.0122347371, 13.31],
     "MILPA ALTA": [13, -80.3076370813, -0.0003960816, -0.0046275519, 0.0404109748, -0.0003385311, 88.83],
     "GUSTAVO A MADERO": [12, 4951.1519849232, -0.0192311166, -0.2230044121, -2.4476907629, 0.0002629555, 24.51],
     "ALVARO OBREGON": [11, 3388.9302909273, -0.0183477195, -0.1446136909, -1.6755320747, 0.00319275, 29.91],
    }

crime_mapper = {1:[1, 5], 2:[6, 10], 3:[11, 15], 4:[16, 20], 5:[21, 25], 6:[26, 30], 7:[31, 35], 8:[36, 40], 9:[41, 45], 10:[46, 50], 11:[51, 55], 12:[56, 60]}

def get_prediction(dia, mes, anio, aqi, alcaldia):
    time = [dia, mes, anio, aqi]
    prediction = models[alcaldia][1]

    for i in range(len(models[alcaldia])-3):
        prediction = prediction + models[alcaldia][i+2]*time[i]

    if(prediction<0):
        prediction = 1
        
    round_prediction = round(((math.modf(prediction)[0])*5%100) + crime_mapper[math.floor(prediction)][0])
    
    return f'{round_prediction} violent crimes, model accuracy: {models[alcaldia][-1]}%'
    
#df_all = pd.read_csv("../CSV_limpios/df_air_crime_evaluation.csv")
df_prepared = df_all.copy()




app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
             html.H1("CRIMEPOLL", style={'text-align': 'center'}),
             html.Br(),
             html.Br()
        ], width=12)
    ]),

    dbc.Row([
        dbc.Col([
            html.Div([
                daq.ToggleSwitch(
                    id='toggle-switch',
                    value=False,
                    label='Switch PM2.5 / PM2.5 with Ozone',
                    labelPosition='bottom'
                ),
                html.Br(),
                html.Br(),
                html.Div(id='toggle-switch-output'),
                html.Br(),
                html.Br()
            ])
        ], width=12)
    ]),

    dbc.Row([
        dbc.Col([
            html.Label('Select year'),
             dcc.Dropdown(df_all["Anio"].unique(), df_all["Anio"].unique()[
                 0], id='slct_year', multi=False, style={'width': "100%", })
        ], width=6),

        dbc.Col([
            html.Label('Select Alcaldia'),
             dcc.Dropdown(id='slct_alcaldia', multi=False, clearable=True,
                    options=[],
                    value=df_all["Alcaldia"].unique()[0],
                    style={'width': "100%"})
        ], width=6)

    ]),

    dbc.Row([
        dbc.Col([
            html.Br(),
            html.Div(id='output_container', children=[]),
            html.Br()
        ], width=6)
    ]),

    dbc.Row([
        dbc.Col([
            html.Br(),
            html.H3('Crime predictor'),
        ], width=12)
    ]),

    dbc.Row([
        dbc.Col([
            html.Br(),
            html.Label('Select date you want to predict:'),
            html.Br(),
            html.Br(),
            dcc.DatePickerSingle(
                id='slct_date',
                min_date_allowed=date(2016, 1, 1),
                max_date_allowed=date(2030, 12, 31),
                initial_visible_month=date.today(),
                date=date.today(),
                display_format='D-M-Y'
            )
        ], width=6),

        dbc.Col([
            html.Br(),
            html.Label('Insert the AQI for prediction:'),
            html.Br(),
            html.Br(),
            daq.NumericInput(
                id='slct_aqi',
                value=70,
                min=5,
                max=180, 
                size=120
            )
        ], width=6)
    ]),

    dbc.Row([
        dbc.Col([
            html.Br(),
            html.Div(id='output_prediction', children=[]),
            html.Br()
        ], width=12)
    ]),

    dbc.Row([
        dbc.Col([
             dcc.Graph(id='plots_air_crime', figure={})
        ], width=12)
    ]), 

    dbc.Row([
        dbc.Col([
            html.Br(),
            html.Br()
        ], width=12)
    ]),

    dbc.Row([
        dbc.Col([
             dcc.Graph(id='air_map', figure={})
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Br(),
            html.Br()
        ], width=12)
    ]),

    dbc.Row([
        dbc.Col([
             dcc.Graph(id='only_air_map', figure={})
        ], width=12)
    ]),

    dbc.Row([
        dbc.Col([
            html.Br(),
            html.Br()
        ], width=12)
    ]),

    dbc.Row([
        dbc.Col([
             dcc.Graph(id='only_crime_map', figure={})
        ], width=12)
    ]),

    dbc.Row([
        dbc.Col([
            html.Br(),
            html.Br(),
            html.Div(
                [
                    dbc.Button("CrimePoll on GitHub", outline=True, color="dark", href="https://github.com/danielgonz77/CrimePoll_v3")
                ],
                className="d-grid gap-2 col-6 mx-auto",
            ),
            html.Br(),
            html.Br()
        ], width=12)
    ])

    # dbc.Row([
    #     dbc.Col([
    #         html.Br(),
    #         html.Br(),
    #         html.H2("Find this project and all csv files in the next link:"),
    #         html.A(html.Button("CrimePoll on GitHub"), href="https://github.com/danielgonz77/CrimePoll_v3"),
    #         #dcc.Link("CrimePoll on GitHub", href="https://github.com/danielgonz77/CrimePoll_v3", style="fontsize:15px"),
    #         html.Br(),
    #         html.Br()
    #     ], width=12)
    # ])

    
])

@app.callback(
    Output('slct_alcaldia', "options"),
    [Input('slct_year', 'value'),
    Input('toggle-switch', 'value')]
)
def get_alcaldias_options(slct_year, dataset_slct):
    if(slct_year == None):
        slct_year = 2016
    
    df_all_o3 = pd.read_csv("./csv/df_air_crime_evaluation_o3.csv")

    if(dataset_slct == True):
        df_prepared = df_all_o3.copy()
    else:
        df_prepared = df_all.copy()
        
    df_alcaldia = df_prepared[df_prepared["Anio"] == slct_year]
    return [{'label': i, 'value': i} for i in df_alcaldia["Alcaldia"].unique()]


@app.callback(
    [Output(component_id='output_container', component_property='children'),
     Output(component_id='output_prediction', component_property='children'),
     Output(component_id='air_map', component_property='figure'),
     Output(component_id='plots_air_crime', component_property='figure'),
     Output(component_id='only_air_map', component_property='figure'),
     Output(component_id='only_crime_map', component_property='figure'),
     Output(component_id='toggle-switch-output', component_property='children')],
    [Input(component_id='slct_year', component_property='value'),
     Input(component_id='slct_alcaldia', component_property='value'),
     Input(component_id='slct_date', component_property='date'),
     Input(component_id='slct_aqi', component_property='value'),
     Input(component_id='toggle-switch', component_property='value')]
)
# below of each callback, need to create a function with args=inputs
def update_graph(option_slctd, alcaldia_slct, date_slct, aqi_slct, dataset_slct):
    print(option_slctd)
    #df_all = pd.read_csv("../CSV_limpios/df_air_crime_evaluation.csv")
    df_all_o3 = pd.read_csv("./csv/df_air_crime_evaluation_o3.csv")

    #df_prepared = df_all.copy()
    if(dataset_slct == True):
        df_prepared = df_all_o3.copy()
        o3 = "PM2.5 with Ozone"
    else:
        df_prepared = df_all.copy()
        o3 = "PM2.5"

    dataset_slct_text = f"You've selected the dataset that uses {o3}."
    
    container = "The year that you've selected was: {} and Alcaldia: {}".format(
        option_slctd, alcaldia_slct, date_slct)


    date_slct = datetime.datetime.strptime(date_slct, '%Y-%m-%d').date()
    prediction = get_prediction(date_slct.day, date_slct.month, date_slct.year, aqi_slct, alcaldia_slct)

    prediction_text = "Date to predict: {} in {} with {} of AQI value is: {}".format(
        date_slct.strftime("%A, %B %d, %Y"), alcaldia_slct, aqi_slct, prediction)

    aire_map_copy = aire_map.copy()
    cimen_map_copy = crime_map.copy()

    aire_map_copy = aire_map[aire_map["Anio"] == option_slctd]
    cimen_map_copy = crime_map[crime_map["Anio"] == option_slctd]

    df_alcaldia = df_prepared.loc[(df_prepared['Anio'] == option_slctd)]

    df_group_by_plot = df_alcaldia.groupby(['Alcaldia', 'Latitud', 'Longitud']).agg(avgAQI=('AQI', 'mean'), sumCrime=('Crimenes', 'sum')).reset_index()


    # Only air
    fig_air = px.scatter_mapbox(aire_map_copy, lat="Latitud", lon="Longitud", hover_name="Alcaldia", hover_data=["Estacion", "avgAQI"], size="avgAQI", size_max=50, color="avgAQI",
                            color_continuous_scale=px.colors.sequential.Jet, zoom=10, height=700)
    fig_air.update_layout(title="Air Pollution Map - {}".format(option_slctd))
    fig_air.update_layout(mapbox_style="open-street-map") #carto-darkmatter
    fig_air.update_layout(margin={"r": 0, "t": 50, "l": 0, "b": 0})
    #fig_air.savefig("Air Pollution Map - {}.pdf".format(option_slctd), bbox_inches='tight')

    # map style : stamen-terrain, carto-positron, open-street-map, carto-darkmatter

    # Only crime
    fig_crime = px.scatter_mapbox(cimen_map_copy, lat="Latitud", lon="Longitud", hover_name="Alcaldia", hover_data=["Estacion", "countCrimenes"], size="countCrimenes", size_max=45, color="countCrimenes",
                            color_continuous_scale=px.colors.sequential.Jet, zoom=10, height=700)
    fig_crime.update_layout(title="Crime Map - {}".format(option_slctd))
    fig_crime.update_layout(mapbox_style="open-street-map")
    fig_crime.update_layout(margin={"r": 0, "t": 50, "l": 0, "b": 0})
    #fig_crime.savefig("Crime Map - {}.pdf".format(option_slctd), bbox_inches='tight')


    # Plotly Express
    fig = px.scatter_mapbox(df_group_by_plot, lat="Latitud", lon="Longitud", hover_name="Alcaldia", hover_data=["avgAQI", "sumCrime"], size="sumCrime", size_max=50, color="avgAQI",
                            color_continuous_scale=px.colors.sequential.Jet, zoom=10, height=700)
    fig.update_layout(title="Map of Relationship between Crime & AQI - {}".format(option_slctd))
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "t": 50, "l": 0, "b": 0})
    #fig.savefig("Map of Relationship between Crime & AQI - {}.pdf".format(option_slctd), bbox_inches='tight')
    # fig.show()

    # Plots
    df_all_plots = df_prepared[(df_prepared['Anio'] == option_slctd) & (df_prepared['Alcaldia'] == alcaldia_slct) & (df_prepared['AQI'] > 5)]

    df_all_plots = df_all_plots.reset_index(drop=True)

    df_all_plots["r"] = round(df_all_plots["r"].mean(), 5)
    df_all_plots["p value"] = round(df_all_plots["p value"].mean(), 5)
    df_all_plots["% / 10 AQI"] = round(df_all_plots["% / 10 AQI"].mean(), 5)


    df_all_plots = df_all_plots.rename(columns={"Crimenes": "Crimes"})
    figure2 = px.scatter(data_frame = df_all_plots, x="AQI",
                    y="Crimes", 
                    trendline="ols", 
                    title="Relationship between Crime & AQI - {} in {}".format(option_slctd, alcaldia_slct),
                    hover_data=["r", "p value", "% / 10 AQI", "ID"])
    
    #figure2.savefig("Relationship between Crime & AQI - {} in {}.pdf".format(option_slctd, alcaldia_slct), bbox_inches='tight')
    
    
    # if len(df_all_plots.AQI) > 2:
    #     pearson_coef, p_value = stats.pearsonr(df_all_plots.AQI, df_all_plots.Crimenes)

    #     df_all_plots["r"] = " {:f}".format(pearson_coef)
    #     #r = (Pearson correlation coefficient)
        
    #     df_all_plots["p value"] = " {:f}".format(p_value)

    #     df_all_fit = np.polyfit(df_all_plots.AQI, df_all_plots.Crimenes, 1)
    #     slope = df_all_fit[0]

    #     percentage = (slope*1000)/df_all_plots["Crimenes"].max()

    #     df_all_plots["% / 10 AQI"] = " {:f}".format(percentage)
    #     #R^2 : The coefficient of determination. A statistical measure of how well the regression line approximates the real data points
    #     #print("For every 10 units increases in AQI, the predicted value of Crime increases by {:.5f} %.".format(percentage))
    # else:
    #     df_all_plots["r"] = " NULL"
    #     df_all_plots["p value"] = " NULL"
    #     df_all_plots["% / 10 AQI"] = " NULL"

    return container, prediction_text, fig, figure2, fig_air, fig_crime, dataset_slct_text





if __name__ == '__main__':
    app.run_server(debug=True)
