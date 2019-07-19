from dash.dependencies import Input, Output
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.decomposition import PCA
import numpy as np
import plotly.graph_objs as go
from sklearn.neighbors import KNeighborsClassifier
from functools import partial
from modAL.batch import uncertainty_batch_sampling
from modAL.models import ActiveLearner
import dash_core_components as dcc
import pandas as pd


def register_callbacks(app):
    @app.callback(Output('scatter', 'figure'),
                  [Input('select-dataset', 'value'),
                  Input('query-batch-size', 'value'),
                  Input('button','n_clicks')
                 ])
    def update_scatter_plot(dataset, batch_size, n_clicks):

        if dataset == 'bc':
            raw_data = load_breast_cancer()
        else:
            raw_data = load_iris()
        df = pd.DataFrame(data=np.c_[raw_data['data'], raw_data['target']],
                          columns=raw_data['feature_names'].tolist() + ['target'])
        x = df[raw_data.feature_names]
        y = df.drop(raw_data.feature_names, axis=1)


        # Define our PCA transformer and fit it onto our raw dataset.
        pca = PCA(n_components=2, random_state=100)
        principals = pca.fit_transform(x)
        df_pca = pd.DataFrame(data=principals, columns=['1', '2'])

        # Randomly choose training examples
        df_train = df.sample(n=3)
        df_train_pca = pd.DataFrame(data=pca.fit(df_train)
                              , columns=['1', '2'])

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
        # # ML model
        # knn = KNeighborsClassifier(n_jobs=-1, n_neighbors=3)
        # # batch sampling
        # preset_batch = partial(uncertainty_batch_sampling, n_instances=3)
        # # AL model
        # learner = ActiveLearner(estimator=knn,
        #                         X_training=x_train,
        #                         y_training=y_train,
        #                         query_strategy=preset_batch)
        # predictions = learner.predict(x_raw)
        # print(" unqueried score", learner.score(x_raw, y_raw))
        # if n_clicks is not None and n_clicks > 0:
        #     query_indices, query_instance = learner.query(x_pool)
        #     selected = pca.transform(query_instance)
        #     data = [go.Scatter(x=transformed_data[:, 0],
        #                        y=transformed_data[:, 1],
        #                        mode='markers',
        #                        name='unlabeled data'),
        #             go.Scatter(x=transformed_data[training_indices, 0],
        #                        y=transformed_data[training_indices, 1],
        #                        mode='markers',
        #                        name='init labeled data' ),
        #             go.Scatter(x=selected[:, 0],
        #                        y=selected[:,1],
        #                        mode='markers',
        #                        name='query'+str(n_clicks))
        #             ]
        fig = go.Figure(data)
       # print(type(x_pool))





        return fig
