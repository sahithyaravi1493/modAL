from dash.dependencies import Input, Output
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.decomposition import PCA
import numpy as np
import plotly.graph_objs as go
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from functools import partial
from modAL.batch import uncertainty_batch_sampling
from modAL.models import ActiveLearner
import dash_core_components as dcc
import pandas as pd

cmap_light = colorscale=[[0.5, "rgb(165,0,38)"],
                [0.45, "rgb(215,48,39)"],
                [0.4, "rgb(244,109,67)"],
                [0.35, "rgb(253,174,97)"],
                [0.3, "rgb(254,224,144)"],
                [0.25, "rgb(224,243,248)"],
                [0.2, "rgb(171,217,233)"],
                [0.15, "rgb(116,173,209)"],
                [0.1, "rgb(69,117,180)"],
                [0.0, "rgb(49,54,149)"]]
cmap_bold = [[0, '#FF0000'], [0.5, '#00FF00'], [1, '#0000FF']]
def register_callbacks(app):
    @app.callback([Output('scatter', 'figure'),
                  Output('decision','figure')],
                  [Input('select-dataset', 'value'),
                  Input('query-batch-size', 'value'),
                  Input('button', 'n_clicks')
                 ])
    def update_scatter_plot(dataset, batch_size, n_clicks):

        if dataset == 'bc':
            raw_data = load_breast_cancer()
        else:
            raw_data = load_iris()
        df = pd.DataFrame(data=np.c_[raw_data['data'], raw_data['target']],
                          columns=raw_data['feature_names'].tolist() + ['target'])

        # Numpy matrices are supported by Active learner, hence use .values
        x = df[raw_data.feature_names].values
        y = df.drop(raw_data.feature_names, axis=1).values

        # Define our PCA transformer and fit it onto our raw dataset.
        pca = PCA(n_components=2, random_state=100)
        principals = pca.fit_transform(x)
        df_pca = pd.DataFrame(data=principals, columns=['1', '2'])

        # Randomly choose training examples
        df_train = df.sample(n=3)
        x_train = df_train[raw_data.feature_names].values
        y_train = df_train.drop(raw_data.feature_names, axis=1).values
        pca_mat = pca.fit_transform(df_train)
        df_train_pca = pd.DataFrame(pca_mat, columns=['1','2'])

        # Unlabeled pool
        data = [go.Scatter(x=df_pca['1'],
                           y=df_pca['2'],
                           mode='markers',
                           name='unlabeled data'),
                go.Scatter(x=df_train_pca['1'],
                           y=df_train_pca['2'],
                           mode='markers',
                           name='labeled data')
                ]
        #
        if n_clicks is None:
            df_pool = df[~df.index.isin(df_train.index)]
            x_pool = df_pool[raw_data['feature_names']].values
            y_pool = df_pool.drop(raw_data['feature_names'], axis=1).values.ravel()
        # ML model
        rf = RandomForestClassifier(n_jobs=-1, n_estimators=20, max_features=0.8)
        # batch sampling
        preset_batch = partial(uncertainty_batch_sampling, n_instances=3)
        # AL model
        learner = ActiveLearner(estimator=rf,
                                X_training=x_train,
                                y_training=y_train.ravel(),
                                query_strategy=preset_batch)
        predictions = learner.predict(x)
        print(" unqueried score", learner.score(x, y.ravel()))
        data_dec = []

        if n_clicks is not None and n_clicks > 0:
            x_pool = np.load('x_pool.npy')
            y_pool = np.load('y_pool.npy')
            query_indices, query_instance, uncertainity = learner.query(x_pool)

            # Plot the query instances
            selected = pca.fit_transform(query_instance)
            data = [go.Scatter(x=df_pca['1'],
                               y=df_pca['2'],
                                mode='markers',
                                name='unlabeled data'),
                    go.Scatter(x=selected[:, 0],
                                y=selected[:, 1],
                                mode='markers',
                                name='query'+str(n_clicks))
                     ]
            # Get the labels for the query instances
            learner.teach(x_pool[query_indices], y_pool[query_indices])
            # Remove query indices from unlabelled pool
            x_pool = np.delete(x_pool, query_indices, axis=0)
            y_pool = np.delete(y_pool, query_indices)
            predictions = learner.predict(x)
            print (uncertainity)

            data_dec = [go.Scatter(x=df_pca['1'],
                               y=df_pca['2'],
                               mode='markers',
                               name='unlabeled data',
                               marker=dict(color=predictions,
                                           colorscale=cmap_bold
                                           )),
                        go.Heatmap(x=df_pca['1'], y=df_pca['2'],
                                   z=uncertainity, colorscale='viridis',
                                   showscale=True)

                    ]

        np.save ('x_pool.npy', x_pool)
        np.save('y_pool.npy', y_pool)
        print('score after query '+str(n_clicks) + ' ' + str(learner.score(x, y)))
        fig = go.Figure(data)
        decision = go.Figure(data_dec)
        return fig, decision
