# Import base data workflow
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# Import Kivy framework
from kivymd.app import MDApp
from kivy.app import App
from kivy.metrics import dp
from kivymd.uix.datatables import MDDataTable
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.label import Label
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.dropdown import DropDown
from kivy.lang import Builder
from kivy.uix.button import Button
from kivy.garden.matplotlib import FigureCanvasKivyAgg

# Import tkinter widgets
import tkinter as tk
from tkinter import Tk
from tkinter import ttk
from tkinter import StringVar
from tkinter import Radiobutton
from tkinter import filedialog

# Import sklearn models and metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVC
from sklearn.svm import LinearSVR

from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score, roc_auc_score, mean_squared_error, mean_absolute_error, \
    mean_squared_log_error, explained_variance_score, roc_curve

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

# Other
import webbrowser

filepath = ''
chosen_model = None
chosen_task = None


class DataLoader(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        boxlayout = BoxLayout(orientation="horizontal", spacing=5, padding=[10])
        floatlayout = FloatLayout()

        button_data_loader = Button(
            text="DataLoader",
            background_color=[0, 1.5, 3, 1],
            size_hint=[1, .05],
            on_press=self.to_dataloader,
        )

        button_data_analysis = Button(
            text="DataAnalysis",
            background_color=[0, 1.5, 3, 1],
            size_hint=[1, .05],
            on_press=self.to_dataanalysis,
        )

        button_data_visualizer = Button(
            text="DataVisualizer",
            background_color=[0, 1.5, 3, 1],
            size_hint=[1, .05],
            on_press=self.to_datavisualizer,
        )

        button_data_ml = Button(
            text="DataML",
            background_color=[0, 1.5, 3, 1],
            size_hint=[1, .05],
            on_press=self.to_dataml
        )

        button_file_path = Button(
            text="Choose file path",
            background_color=[0, 1.5, 3, 1],
            size_hint=[.2, .1],
            on_press=self.file_path,
            pos=(20, 400)
        )

        button_upload_data = Button(
            text="Upload data",
            background_color=[0, 1.5, 3, 1],
            size_hint=[.2, .1],
            on_press=self.upload_data,
            pos=(20, 300)
        )

        """button_to_github = Button(
            pos=(20, 700),
            background_normal="C:/Users/Mr0Wo/PycharmProjects/pythonProject/icons/github.png",
            size_hint=[.1, .1],
            on_press=self.to_github
        )"""

        boxlayout.add_widget(button_data_loader)
        boxlayout.add_widget(button_data_analysis)
        boxlayout.add_widget(button_data_visualizer)
        boxlayout.add_widget(button_data_ml)
        floatlayout.add_widget(boxlayout)
        floatlayout.add_widget(button_file_path)
        floatlayout.add_widget(button_upload_data)
        # floatlayout.add_widget(button_to_github)
        self.add_widget(floatlayout)

    def to_dataloader(self, *args):
        self.manager.transition.direction = 'left'
        self.manager.current = 'DataLoader'

    def to_dataanalysis(self, *args):
        self.manager.transition.direction = 'left'
        self.manager.current = 'DataAnalysis'

    def to_datavisualizer(self, *args):
        self.manager.transition.direction = 'left'
        self.manager.current = 'DataVisualizer'

    def to_dataml(self, *args):
        self.manager.transition.direction = 'left'
        self.manager.current = 'DataML'


    def file_path(self, *args):
        global filepath
        filepath = filedialog.askopenfilename()
        if filepath != '':
            message = "Выбранный путь к файлу: " + filepath
            tk.messagebox.showinfo(title="Файл выбран", message=message)
            filepath = filepath
        else:
            message_of_decline = "Путь к файлу не выбран"
            tk.messagebox.showinfo(title="Файл не выбран", message=message_of_decline)
            filepath = ''

    def upload_data(self, *args):
        global df
        # global columns
        # global values
        if filepath != '':
            file_type = filepath[filepath.rfind('.') + 1:]
            if file_type == 'xlsx' or file_type == 'xls':
                df = pd.read_excel(filepath)
            elif file_type == 'csv':
                df = pd.read_csv(filepath)
            elif file_type == 'json':
                df = pd.read_json(filepath)

            """
            columns = df.columns.values
            values = df.values
            data_tables = MDDataTable(
                size_hint=(.7, .6),
                use_pagination=True,
                pos_hint={'center_x': .6, 'center_y': .5},
                column_data=[
                    (col, dp(30))
                    for col in columns
                ],
                row_data=values
            )
            self.add_widget(data_tables)
            """
        else:
            pass

    # def to_github(self, *args):
    #     webbrowser.open("https://github.com/Mr0Wolfy")


class DataAnalysis(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        boxlayout = BoxLayout(orientation="horizontal", spacing=5, padding=[10])
        floatlayout = FloatLayout()

        button_data_loader = Button(
            text="DataLoader",
            background_color=[0, 1.5, 3, 1],
            size_hint=[1, .05],
            on_press=self.to_dataloader,
        )

        button_data_analysis = Button(
            text="DataAnalysis",
            background_color=[0, 1.5, 3, 1],
            size_hint=[1, .05],
            on_press=self.to_dataanalysis,
        )

        button_data_visualizer = Button(
            text="DataVisualizer",
            background_color=[0, 1.5, 3, 1],
            size_hint=[1, .05],
            on_press=self.to_datavisualizer,
        )

        button_data_ml = Button(
            text="DataML",
            background_color=[0, 1.5, 3, 1],
            size_hint=[1, .05],
            on_press=self.to_dataml
        )

        button_EDA = Button(
            text='Первичный анализ',
            background_color=[0, 1.5, 3, 1],
            size_hint=[.2, .2],
            pos=(20, 300),
            on_press=self.EDA
        )

        boxlayout.add_widget(button_data_loader)
        boxlayout.add_widget(button_data_analysis)
        boxlayout.add_widget(button_data_visualizer)
        boxlayout.add_widget(button_data_ml)
        floatlayout.add_widget(boxlayout)
        floatlayout.add_widget(button_EDA)
        self.add_widget(floatlayout)

    def to_dataloader(self, *args):
        self.manager.transition.direction = 'right'
        self.manager.current = 'DataLoader'

    def to_dataanalysis(self, *args):
        self.manager.transition.direction = 'left'
        self.manager.current = 'DataAnalysis'

    def to_datavisualizer(self, *args):
        self.manager.transition.direction = 'left'
        self.manager.current = 'DataVisualizer'

    def to_dataml(self, *args):
        self.manager.transition.direction = 'left'
        self.manager.current = 'DataML'


    def EDA(self, *args):
        if filepath != '':
            output = df.dtypes.reset_index().rename(columns={'index': 'Column names', 0: 'Column types'})
            output = output.merge(df.isna().sum().reset_index().rename(columns={'index': 'Column names', 0: 'Nan values'}))
            output['% of nan values'] = np.round((output['Nan values'] / df.shape[0]), 2)
            output = output.merge(df.describe().T.reset_index().rename(columns={'index': 'Column names'}),how='left').fillna('-')
            columns_EDA = output.columns.values
            values_EDA = output.values


            EDA_table = MDDataTable(
                    size_hint=(.7, .6),
                    pos_hint={'center_x': .6, 'center_y': .5},
                    use_pagination=True,
                    column_data=[
                        (col, dp(30))
                        for col in columns_EDA
                    ],
                    row_data=values_EDA
                )
            self.add_widget(EDA_table)
        else:
            pass

class uiApp(App):

    def build(self, *args):
        self.str = Builder.load_string(""" 

BoxLayout:
    layout:layout

    BoxLayout:

        id:layout

                                """)

        signal = [7, 89.6, 45. - 56.34]

        signal = np.array(signal)

        # this will plot the signal on graph
        plt.plot(signal)

        # setting x label
        plt.xlabel('Time(s)')

        # setting y label
        plt.ylabel('signal (norm)')
        plt.grid(True, color='lightgray')

        # adding plot to kivy boxlayout
        self.str.layout.add_widget(FigureCanvasKivyAgg(plt.gcf()))
        return self.str

class DataVisualizer(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        boxlayout = BoxLayout(orientation="horizontal", spacing=5, padding=[10])
        floatlayout = FloatLayout()


        button_data_loader = Button(
            text="DataLoader",
            background_color=[0, 1.5, 3, 1],
            size_hint=[1, .05],
            on_press=self.to_dataloader,
        )

        button_data_analysis = Button(
            text="DataAnalysis",
            background_color=[0, 1.5, 3, 1],
            size_hint=[1, .05],
            on_press=self.to_dataanalysis,
        )

        button_data_visualizer = Button(
            text="DataVisualizer",
            background_color=[0, 1.5, 3, 1],
            size_hint=[1, .05],
            on_press=self.to_datavisualizer,
        )

        button_data_ml = Button(
            text="DataML",
            background_color=[0, 1.5, 3, 1],
            size_hint=[1, .05],
            on_press=self.to_dataml
        )

        button_choose_columns = Button(
            text="Выбрать колонки",
            background_color=[0, 1.5, 3, 1],
            size_hint=[.2, .1],
            on_press=self.choose_columns,
            pos=(70, 600)
        )

        button_create_graph = Button(
            text="Построить график",
            background_color=[0, 1.5, 3, 1],
            size_hint=[.2, .1],
            on_press=self.create_graph,
            pos=(530, 600)
        )

        dropdown = DropDown()
        graphs = ['scatterplot', 'barplot', 'lineplot', 'boxplot']
        for graph in graphs:
            # Adding button in drop down list
            btn = Button(text='%s' % graph, size_hint_y=None, height=40)

            # binding the button to show the text when selected
            btn.bind(on_release=lambda btn: dropdown.select(btn.text))

            # then add the button inside the dropdown
            dropdown.add_widget(btn)

        button_drop_down_graphs = Button(
            text='Выберите график',
            size_hint=[.2, .1],
            background_color=[0, 1.5, 3, 1],
            pos=(300, 600)
        )
        button_drop_down_graphs.bind(on_release=dropdown.open)
        dropdown.bind(on_select=lambda instance, x: setattr(button_drop_down_graphs, 'text', x))


        boxlayout.add_widget(button_data_loader)
        boxlayout.add_widget(button_data_analysis)
        boxlayout.add_widget(button_data_visualizer)
        boxlayout.add_widget(button_data_ml)
        floatlayout.add_widget(boxlayout)
        floatlayout.add_widget(button_create_graph)
        floatlayout.add_widget(button_choose_columns)
        floatlayout.add_widget(button_drop_down_graphs)
        self.add_widget(floatlayout)

    def to_dataloader(self, *args):
        self.manager.transition.direction = 'right'
        self.manager.current = 'DataLoader'

    def to_dataanalysis(self, *args):
        self.manager.transition.direction = 'right'
        self.manager.current = 'DataAnalysis'

    def to_datavisualizer(self, *args):
        self.manager.transition.direction = 'right'
        self.manager.current = 'DataVisualizer'

    def to_dataml(self, *args):
        self.manager.transition.direction = 'left'
        self.manager.current = 'DataML'


    def choose_columns(self, *args):
        if filepath != '':
            choose_columns_values = df.columns.values.reshape(-1, 1)
            choose_columns_table = MDDataTable(
                size_hint=(.3, .6),
                use_pagination=True,
                check=True,
                pos=(70, 90),
                column_data=
                [
                    ('Name', dp(30))

                ]
                ,
                row_data=choose_columns_values
            )
            self.remove_widget(choose_columns_table)
            self.add_widget(choose_columns_table)
        else:
            pass

    def create_graph(self, *args):
        uiApp().run()

class DataML(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        global chosen_model

        change_boxlayout = BoxLayout(orientation="horizontal", spacing=5, padding=[10])
        floatlayout = FloatLayout()

        label_choose = Label(
            text='[color=000000]Вставьте название целевой переменной:[/color]',
            size_hint=[.1, .1],
            pos=(150, 650),
            markup=True
        )

        global user_label_input
        user_label_input = TextInput(
            size_hint=[.1, .05],
            pos=(400, 670)
        )

        button_data_loader = Button(
            text="DataLoader",
            background_color=[0, 1.5, 3, 1],
            size_hint=[1, .05],
            on_press=self.to_dataloader
        )

        button_data_analysis = Button(
            text="DataAnalysis",
            background_color=[0, 1.5, 3, 1],
            size_hint=[1, .05],
            on_press=self.to_dataanalysis
        )

        button_data_visualizer = Button(
            text="DataVisualizer",
            background_color=[0, 1.5, 3, 1],
            size_hint=[1, .05],
            on_press=self.to_datavisualizer
        )

        button_data_ml = Button(
            text="DataML",
            background_color=[0, 1.5, 3, 1],
            size_hint=[1, .05],
            on_press=self.to_dataml
        )

        dropdown_model = DropDown()
        models = [
            'Решающие деревья',
            'Линейная регрессия',
            'Логистическая регрессия',
            'Опорные вектора',
            'Ближайшие соседи',
            'Случайный лес'
                  ]
        for model in models:
            # Adding button in drop down list
            btn = Button(text='%s' % model, size_hint_y=None, height=40)

            # binding the button to show the text when selected
            btn.bind(on_release=lambda btn: dropdown_model.select(btn.text))

            # then add the button inside the dropdown
            dropdown_model.add_widget(btn)

        global button_drop_down_models
        button_drop_down_models = Button(
            text='Выберите модель',
            size_hint=[.23, .05],
            background_color=[0, 1.5, 3, 1],
            pos=(25, 600),

        )
        button_drop_down_models.bind(on_release=dropdown_model.open)
        dropdown_model.bind(on_select=self.drop_down_choose_model_bind)


        dropdown_task = DropDown()
        tasks = [
            'Регрессия',
            'Классификация',
            'Кластеризация',
        ]
        for task in tasks:
            # Adding button in drop down list
            btn = Button(text='%s' % task, size_hint_y=None, height=40)

            # binding the button to show the text when selected
            btn.bind(on_release=lambda btn: dropdown_task.select(btn.text))

            # then add the button inside the dropdown
            dropdown_task.add_widget(btn)

        global button_drop_down_tasks
        button_drop_down_tasks = Button(
            text='Выберите задачу',
            size_hint=[.23, .05],
            background_color=[0, 1.5, 3, 1],
            pos=(270, 600),

        )
        button_drop_down_tasks.bind(on_release=dropdown_task.open)
        dropdown_task.bind(on_select=self.drop_down_choose_model_tasks_bind)



        button_model_params = Button(
            text='Параметры модели',
            size_hint=[.23, .05],
            background_color=[0, 1.5, 3, 1],
            pos=(25, 550),
            on_press=self.model_params
        )

        button_model_train = Button(
            text='Обучить модель',
            size_hint=[.23, .05],
            background_color=[0, 1.5, 3, 1],
            pos=(25, 500),
            on_press=self.ml_coach
        )

        button_model_metrics = Button(
            text='Метрики модели',
            size_hint=[.23, .05],
            background_color=[0, 1.5, 3, 1],
            pos=(270, 550),
            on_press=self.model_metrics
        )

        dropdown_ml_graphs = DropDown()
        ml_graphs = [
            'Матрица ошибок',
            'Рок кривая',
        ]
        for graph in ml_graphs:
            # Adding button in drop down list
            btn = Button(text='%s' % graph, size_hint_y=None, height=40)

            # binding the button to show the text when selected
            btn.bind(on_release=lambda btn: dropdown_ml_graphs.select(btn.text))

            # then add the button inside the dropdown
            dropdown_ml_graphs.add_widget(btn)

        global button_drop_down_ml_graphs
        button_drop_down_ml_graphs = Button(
            text='Выберите график',
            size_hint=[.23, .05],
            background_color=[0, 1.5, 3, 1],
            pos=(600, 600),

        )
        button_drop_down_ml_graphs.bind(on_release=dropdown_ml_graphs.open)
        dropdown_ml_graphs.bind(on_select=self.drop_down_choose_ml_graphs_bind)


        button_ml_graphs = Button(
            text='Построить график',
            size_hint=[.23, .05],
            background_color=[0, 1.5, 3, 1],
            pos=(600, 550),
            on_press=self.get_ml_graphs
        )


        change_boxlayout.add_widget(button_data_loader)
        change_boxlayout.add_widget(button_data_analysis)
        change_boxlayout.add_widget(button_data_visualizer)
        change_boxlayout.add_widget(button_data_ml)

        floatlayout.add_widget(label_choose)
        floatlayout.add_widget(user_label_input)
        floatlayout.add_widget(button_drop_down_models)
        floatlayout.add_widget(button_model_train)
        floatlayout.add_widget(button_model_params)
        floatlayout.add_widget(button_drop_down_tasks)
        floatlayout.add_widget(button_model_metrics)
        floatlayout.add_widget(button_drop_down_ml_graphs)
        floatlayout.add_widget(button_ml_graphs)

        floatlayout.add_widget(change_boxlayout)


        self.add_widget(floatlayout)

    def to_dataloader(self, *args):
        self.manager.transition.direction = 'right'
        self.manager.current = 'DataLoader'

    def to_dataanalysis(self, *args):
        self.manager.transition.direction = 'right'
        self.manager.current = 'DataAnalysis'

    def to_datavisualizer(self, *args):
        self.manager.transition.direction = 'right'
        self.manager.current = 'DataVisualizer'
    def to_dataml(self, *args):
        self.manager.transition.direction = 'left'
        self.manager.current = 'DataML'

    def drop_down_choose_model_bind(self, instance, x):
        global chosen_model
        chosen_model = x
        return setattr(button_drop_down_models, 'text', x)

    def drop_down_choose_model_tasks_bind(self, instance, x):
        global chosen_task
        chosen_task = x
        return setattr(button_drop_down_tasks, 'text', x)

    def drop_down_choose_ml_graphs_bind(self, instance, x):
        global chosen_ml_graphs
        chosen_ml_graphs = x
        return setattr(button_drop_down_ml_graphs, 'text', x)

    def get_ml_graphs(self, *args):
        if chosen_ml_graphs == 'Матрица ошибок':
            plt.xlabel("Y_pred")
            plt.ylabel("Y_true")
            plt.title("Матрица ошибок")
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
            plt.show()
        elif chosen_ml_graphs == 'Рок кривая':
            y_pred_proba = model.predict_proba(X_test)[::, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            plt.xlabel("False positive rate")
            plt.ylabel("True positive rate")
            plt.title("Рок кривая")
            plt.plot(fpr, tpr)
            plt.show()





    def ml_coach(self, *args):

        if 'df' not in globals():
            tk.messagebox.showinfo(title="Ошибка!",
                                   message='Данные для обучения модели не загруженны')
            pass

        elif user_label_input.text == '':
            tk.messagebox.showinfo(title="Ошибка!",
                                   message='Не выбрана целевая переменная')
            pass

        elif 'flag_basic_params' not in globals() and \
                'flag_optimal_params' not in globals() and \
                'flag_user_params' not in globals():
            tk.messagebox.showinfo(title="Ошибка!",
                                   message='Не заданы параметы модели')
            pass


        else:
            global y_test
            global y_pred
            global X_test
            global model
            label = user_label_input.text
            X = df.drop(columns=label, axis=1)
            y = df[label]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)
            if 'flag_basic_params' in globals():

                if chosen_model == 'Решающие деревья':
                    if chosen_task == 'Классификация':
                        model = DecisionTreeClassifier()
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)

                    else:
                        model = DecisionTreeRegressor()
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)


                elif chosen_model == 'Логистическая регрессия':
                    model = LogisticRegression()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)



                elif chosen_model == 'Линейная регрессия':
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                elif chosen_model == 'Опорные вектора':
                    if chosen_task == 'Классификация':
                        model = LinearSVC()
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)


                    else:
                        model = LinearSVR()
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)


                elif chosen_model == 'Случайный лес':
                    if chosen_task == 'Классификация':
                        model = RandomForestClassifier()
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)


                    else:
                        model = RandomForestRegressor()
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)


                elif chosen_model == 'Ближайшие соседи':
                    if chosen_task == 'Классификация':
                        model = KNeighborsClassifier()
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)


                    elif chosen_task == 'Регрессия':
                        model = KNeighborsRegressor()
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)

                tk.messagebox.showinfo(title="%s" % chosen_model,
                                       message='Моедль успешно обучена!')


            elif 'flag_user_params' in globals():
                if chosen_model == 'Решающие деревья':

                    if chosen_task == 'Классификация':
                        model = DecisionTreeClassifier(criterion=tree_params['criterion'],
                                                       max_depth=tree_params['max_depth'],
                                                       min_samples_split=tree_params['min_samples_split'])
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)



                    else:
                        model = DecisionTreeRegressor(criterion=tree_params['criterion'],
                                                       max_depth=tree_params['max_depth'],
                                                       min_samples_split=tree_params['min_samples_split'])
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)

                    tk.messagebox.showinfo(title="%s" % chosen_model,
                                           message='Моедль успешно обучена!')

                elif chosen_model == 'Случайный лес':
                    if chosen_task == 'Классификация':
                        model = RandomForestClassifier(
                            n_estimators=random_forest_params['n_estimators'],
                            max_depth=random_forest_params['max_depth'],
                            criterion=random_forest_params['criterion'],
                            min_samples_split=random_forest_params['min_samples_split'],
                            min_samples_leaf=random_forest_params['min_samples_leaf']
                        )
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                    else:
                        model = RandomForestRegressor(
                            n_estimators=random_forest_params['n_estimators'],
                            max_depth=random_forest_params['max_depth'],
                            criterion=random_forest_params['criterion'],
                            min_samples_split=random_forest_params['min_samples_split'],
                            min_samples_leaf=random_forest_params['min_samples_leaf']
                        )
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)

                    tk.messagebox.showinfo(title="%s" % chosen_model,
                                           message='Моедль успешно обучена!')

            elif 'flag_optimal_parameters' in globals():
                if chosen_model == 'Решающие деревья':
                    if chosen_task == 'Классификация':
                        params = {'max_depth':range(4, 26, 2),
                                  'min_samples_split':range(1, 10),
                                  'min_samples_leaf': range(1, 10)}
                        clf = DecisionTreeClassifier()
                        grid = GridSearchCV(clf, param_grid=params, cv=5)
                        grid.fit(X_train, y_train)
                        model = grid.best_estimator_
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        tk.messagebox.showinfo(title="%s" % chosen_model,
                                               message='precision: {}'.format(precision_score(y_test, y_pred)))
                    else:
                        pass

                elif chosen_model == 'Линейная регрессия':
                    pass


    def model_params(self, *args):


        def save_user_params(*args):
            global flag_user_params
            flag_user_params = 1
            if chosen_model == 'Решающие деревья':
                global tree_params
                tree_params = {}
                if chosen_task == 'Классификация':
                    tree_params['criterion'] = tree_class_var.get()
                else:
                    tree_params['criterion'] = tree_reg_var.get()
                tree_params['max_depth'] = text_input_max_depth.get()
                tree_params['min_samples_split'] = text_input_min_samples_split.get()

                if (int(text_input_min_samples_split.get()) >= 2) and (int(text_input_max_depth.get()) > 0):
                    tk.messagebox.showinfo(title="Устанока параметров модели",
                                           message='Параметры установлены!')
                    root.destroy()

                elif int(text_input_min_samples_split.get()) < 2:
                    tk.messagebox.showinfo(title="Ошибка!",
                                           message='Параметр min_samples_split должен находиться в полуинтервале [2; +inf)')

                elif int(text_input_max_depth.get()) <= 0:
                    tk.messagebox.showinfo(title="Ошибка!",
                                           message='Параметр max_depth должен находиться в полуинтервале [1; +inf)')

                for key, value in zip(tree_params.keys(), tree_params.values()):
                    if value == "":
                        tree_params[key] = None
                    elif value.isdigit():
                        tree_params[key] = int(value)


            elif chosen_model == 'Случайный лес':
                global random_forest_params
                random_forest_params = {}
                if chosen_task == 'Классификация':
                    random_forest_params['criterion'] = random_forest_class_var.get()
                else:
                    random_forest_params['criterion'] = random_forest_reg_var.get()
                random_forest_params['max_depth'] = text_input_max_depth_r_forest.get()
                random_forest_params['n_estimators'] = text_input_n_estimators_r_forest.get()
                random_forest_params['min_samples_split'] = text_input_min_samples_split_r_forest.get()
                random_forest_params['min_samples_leaf'] = text_input_min_samples_leaf_r_forest.get()

                if (int(text_input_max_depth_r_forest.get()) > 0) and (int(text_input_min_samples_leaf_r_forest.get()) > 0) \
                        and (int(text_input_min_samples_split_r_forest.get()) >= 2) and (int(text_input_n_estimators_r_forest.get()) >= 1):
                    tk.messagebox.showinfo(title="Устанока параметров модели",
                                           message='Параметры установлены!')
                    root.destroy()

                elif int(text_input_n_estimators_r_forest.get()) < 1:
                    tk.messagebox.showinfo(title="Ошибка!",
                                           message='Кол-во оценщиков должен находиться в полуинтервале [1; +inf)')

                elif int(text_input_max_depth_r_forest.get()) <= 0:
                    tk.messagebox.showinfo(title="Ошибка!",
                                           message='Параметр максимальная глубина должен находиться в полуинтервале [1; +inf)')

                elif int(text_input_min_samples_leaf_r_forest.get()) <= 0:
                    tk.messagebox.showinfo(title="Ошибка!",
                                           message='Параметр минимальное кол-во элементов в листе должен находиться в полуинтервале [1; +inf)')

                elif int(text_input_min_samples_split_r_forest.get()) < 2:
                    tk.messagebox.showinfo(title="Ошибка!",
                                           message='Параметр минимальное кол-во элементов для разбиения должен находиться в полуинтервале [2; +inf)')



                for key, value in zip(random_forest_params.keys(), random_forest_params.values()):
                    if value == "":
                        random_forest_params[key] = None
                    elif value.isdigit():
                        random_forest_params[key] = int(value)







        def set_basic_params(*args):
            global flag_basic_params
            flag_basic_params = 1
            tk.messagebox.showinfo(title="Устанока параметров модели", message='Параметры установлены!')
            if 'flag_optimal_parameters' in globals():
                del flag_optimal_parameters
            root.destroy()

        def set_optimal_params(*args):
            global flag_optimal_parameters
            flag_optimal_parameters = 1
            tk.messagebox.showinfo(title="Устанока параметров модели", message='Параметры будут подобраны автоматически!')
            if 'flag_basic_params' in globals():
                del flag_basic_params
            root.destroy()



        my_dict = {'Решающие деревья': ['Регрессия', 'Классификация'],
                   'Линейная регрессия': 'Регрессия',
                   'Логистическая регрессия': "Классификация",
                   'Опорные вектора': ["Регрессия", 'Классификация'],
                   'Ближайшие соседи': ['Классификация', 'Кластеризация', 'Регрессия'],
                   'Случайный лес': ["Регрессия", 'Классификация'],
                   'Градиентный бустинг': ["Регрессия", 'Классификация']}

        if chosen_model is not None and chosen_task is not None:
            if chosen_task in my_dict[chosen_model]:

                root = Tk()
                root.resizable(False, False)
                root.title(chosen_model)
                root.geometry("800x350")

                button_save_param = tk.Button(root, text="Сохранить параметры", command=save_user_params)
                button_save_param.pack()
                button_save_param.place(x=20, y=200)

                button_basic_param = tk.Button(root, text="Параметры по умолчанию", command=set_basic_params)
                button_basic_param.pack()
                button_basic_param.place(x=20, y=250)

                button_auto_param = tk.Button(root, text="Автоподбор параметров", command=set_optimal_params)
                button_auto_param.pack()
                button_auto_param.place(x=20, y=300)

                if chosen_model == 'Решающие деревья':
                    label_criterion = tk.Label(text='Выберите критейрий: ')
                    label_criterion.pack()
                    label_criterion.place(x=20, y=30)

                    label_max_depth = tk.Label(text='Максимальная глубина дерева: ')
                    label_max_depth.pack()
                    label_max_depth.place(x=20, y=60)

                    label_min_samples_split = tk.Label(text='Минимальноe кол-во элементов в листе: ')
                    label_min_samples_split.pack()
                    label_min_samples_split.place(x=20, y=90)

                    text_input_max_depth = ttk.Entry(width=10)
                    text_input_max_depth.pack()
                    text_input_max_depth.place(x=250,y=60)

                    text_input_min_samples_split = ttk.Entry(width=10)
                    text_input_min_samples_split.pack()
                    text_input_min_samples_split.place(x=315,y=90)

                    if chosen_task == 'Классификация':
                        tree_class_var = StringVar()
                        tree_class_var.set('gini')
                        r1 = Radiobutton(text='gini',
                                         variable=tree_class_var, value='gini')
                        r2 = Radiobutton(text='entropy',
                                         variable=tree_class_var, value='entropy')

                        r1.pack()
                        r1.place(x=190, y=30)
                        r2.pack()
                        r2.place(x=250, y=30)
                        root.mainloop()

                    elif chosen_task == 'Регрессия':
                        tree_reg_var = StringVar()
                        tree_reg_var.set('squared_error')
                        r1 = Radiobutton(text='squared_error',
                                         variable=tree_reg_var, value='squared_error')
                        r2 = Radiobutton(text='absolute_error',
                                         variable=tree_reg_var, value='absolute_error')
                        r3 = Radiobutton(text='Mean Absolute Error',
                                         variable=tree_reg_var, value='Mean Absolute Error')

                        r1.pack()
                        r1.place(x=190, y=30)
                        r2.pack()
                        r2.place(x=320, y=30)
                        r3.pack()
                        r3.place(x=450, y=30)
                        root.mainloop()


                elif chosen_model == 'Линейная регрессия':
                    root.mainloop()

                elif chosen_model == 'Логистическая регрессия':

                    label_penalty = tk.Label(text='Вид регуляризации: ')
                    label_penalty.pack()
                    label_penalty.place(x=20, y=20)

                    log_reg_var = StringVar()
                    log_reg_var.set('L2')
                    r1 = Radiobutton(text='L1',
                                     variable=log_reg_var, value='L1')
                    r2 = Radiobutton(text='L2',
                                     variable=log_reg_var, value='L2')
                    r3 = Radiobutton(text='Elasticnet',
                                     variable=log_reg_var, value='Elasticnet')

                    r1.pack()
                    r1.place(x=170, y=20)
                    r2.pack()
                    r2.place(x=220, y=20)
                    r3.pack()
                    r3.place(x=270, y=20)
                    root.mainloop()



                elif chosen_model == 'Опорные вектора':

                    label_kernel_svm = tk.Label(text='Ядро алгоритма: ')
                    label_kernel_svm.pack()
                    label_kernel_svm.place(x=20,y=20)

                    kernel_svm_var = StringVar()
                    kernel_svm_var.set('rbf')
                    r1 = Radiobutton(text='linear',
                                     variable=kernel_svm_var, value='linear')
                    r2 = Radiobutton(text='poly',
                                     variable=kernel_svm_var, value='poly')
                    r3 = Radiobutton(text='rbf',
                                     variable=kernel_svm_var, value='rbf')
                    r4 = Radiobutton(text='sigmoid',
                                     variable=kernel_svm_var, value='sigmoid')
                    r5 = Radiobutton(text='precomputed',
                                     variable=kernel_svm_var, value='precomputed')

                    r1.pack()
                    r1.place(x=140, y=20)
                    r2.pack()
                    r2.place(x=210, y=20)
                    r3.pack()
                    r3.place(x=270, y=20)
                    r4.pack()
                    r4.place(x=320, y=20)
                    r5.pack()
                    r5.place(x=400, y=20)

                    root.mainloop()

                elif chosen_model == 'Ближайшие соседи':

                    label_n_neighbors = tk.Label(text='Кол-во ближайших соседей: ')
                    label_n_neighbors.pack()
                    label_n_neighbors.place(x=20, y=20)

                    text_input_n_neighbors = ttk.Entry()
                    text_input_n_neighbors.pack()
                    text_input_n_neighbors.place(x=230, y=20)

                    label_algorithm_neighbors = tk.Label(text='Алгоритм: ')
                    label_algorithm_neighbors.pack()
                    label_algorithm_neighbors.place(x=20, y=60)

                    n_neighbors_var = StringVar()
                    n_neighbors_var.set('auto')
                    r1 = Radiobutton(text='auto',
                                     variable=n_neighbors_var, value='auto')
                    r2 = Radiobutton(text='ball_tree',
                                     variable=n_neighbors_var, value='ball_tree')
                    r3 = Radiobutton(text='kd_tree',
                                     variable=n_neighbors_var, value='kd_tree')
                    r4 = Radiobutton(text='brute',
                                     variable=n_neighbors_var, value='brute')

                    r1.pack()
                    r1.place(x=100, y=60)
                    r2.pack()
                    r2.place(x=155, y=60)
                    r3.pack()
                    r3.place(x=238, y=60)
                    r4.pack()
                    r4.place(x=315, y=60)
                    root.mainloop()

                    if chosen_task == 'Классификация':
                        root.mainloop()
                    elif chosen_task == 'Регрессия':
                        root.mainloop()
                    else:
                        root.mainloop()


                elif chosen_model == 'Случайный лес':

                    label_criterion_r_forest = tk.Label(text='Выберите критейрий: ')
                    label_criterion_r_forest.pack()
                    label_criterion_r_forest.place(x=20, y=20)

                    label_n_estimators_r_forest = tk.Label(text='Количество деревьев: ')
                    label_n_estimators_r_forest.pack()
                    label_n_estimators_r_forest.place(x=20, y=50)

                    text_input_n_estimators_r_forest = ttk.Entry(width=10)
                    text_input_n_estimators_r_forest.pack()
                    text_input_n_estimators_r_forest.place(x=190, y=50)

                    label_max_depth_r_forest = tk.Label(text='Максимальная глубина деревьев: ')
                    label_max_depth_r_forest.pack()
                    label_max_depth_r_forest.place(x=20, y=80)

                    text_input_max_depth_r_forest = ttk.Entry()
                    text_input_max_depth_r_forest.pack()
                    text_input_max_depth_r_forest.place(x=270, y=80)

                    label_min_samples_leaf_r_forest = tk.Label(text='Минимальное кол-во элементов в листе: ')
                    label_min_samples_leaf_r_forest.pack()
                    label_min_samples_leaf_r_forest.place(x=20, y=110)

                    text_input_min_samples_leaf_r_forest = ttk.Entry()
                    text_input_min_samples_leaf_r_forest.pack()
                    text_input_min_samples_leaf_r_forest.place(x=315, y=110)

                    label_min_samples_split_r_forest = tk.Label(text='Минимальное кол-во элементов для разделения: ')
                    label_min_samples_split_r_forest.pack()
                    label_min_samples_split_r_forest.place(x=20, y=140)

                    text_input_min_samples_split_r_forest = ttk.Entry()
                    text_input_min_samples_split_r_forest.pack()
                    text_input_min_samples_split_r_forest.place(x=375, y=140)

                    if chosen_task == 'Классификация':
                        random_forest_class_var = StringVar()
                        random_forest_class_var.set('gini')
                        r1 = Radiobutton(text='gini',
                                         variable=random_forest_class_var, value='gini')
                        r2 = Radiobutton(text='entropy',
                                         variable=random_forest_class_var, value='entropy')
                        r3 = Radiobutton(text='logloss',
                                         variable=random_forest_class_var, value='logloss')

                        r1.pack()
                        r1.place(x=190, y=20)
                        r2.pack()
                        r2.place(x=250, y=20)
                        r3.pack()
                        r3.place(x=330, y=20)

                        root.mainloop()

                    else:
                        random_forest_reg_var = StringVar()
                        random_forest_reg_var.set('squared_error')
                        r1 = Radiobutton(text='squared_error',
                                         variable=random_forest_reg_var, value='squared_error')
                        r2 = Radiobutton(text='absolute_error',
                                         variable=random_forest_reg_var, value='absolute_error')
                        r3 = Radiobutton(text='friedman_mse',
                                         variable=random_forest_reg_var, value='friedman_mse')
                        r4 = Radiobutton(text='poisson',
                                         variable=random_forest_reg_var, value='poisson')

                        r1.pack()
                        r1.place(x=190, y=20)
                        r2.pack()
                        r2.place(x=310, y=20)
                        r3.pack()
                        r3.place(x=435, y=20)
                        r4.pack()
                        r4.place(x=565, y=20)

                        root.mainloop()

                elif chosen_model == 'Градиентный бустинг':

                    root.mainloop()

            else:
                tk.messagebox.showinfo(title="Ошибка!", message='Модель {} не может решать задачи типа {}'.format(chosen_model.lower(), chosen_task.lower()))

        else:
            tk.messagebox.showinfo(title="Ошибка!", message='Пожалуйста, выберите тип модели и задачи')


    def model_metrics(self, *args):
        if 'model' in globals():
            if chosen_task == 'Классификация':
                tk.messagebox.showinfo(title="Метрики качества модели {}".format(chosen_model),
                                    message='Accuracy score: %.2f\n'
                                            'Precision score: %.2f\n'
                                            'Recall score: %.2f\n'
                                            'f1 score: %.2f\n'
                                            'Roc_auc score: %.2f' % (accuracy_score(y_test, y_pred), precision_score(y_test, y_pred),
                                                                    recall_score(y_test, y_pred), f1_score(y_test, y_pred),
                                                                    roc_auc_score(y_test, y_pred)))

            elif chosen_task == 'Регрессия':
                tk.messagebox.showinfo(title="Метрики качества модели {}".format(chosen_model),
                                       message='Mean_squared_error: %.2f\n'
                                               'Mean_absolute_error: %.2f\n'
                                               'Explained_variance_score: %.2f'
                                       % (mean_squared_error(y_test, y_pred),
                                          mean_absolute_error(y_test, y_pred),
                                          explained_variance_score(y_test, y_pred)))
        else:
            tk.messagebox.showinfo(title="Ошибка!",
                                   message='Модель еще не обучена')

class DataLabApp(MDApp):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(DataLoader(name='DataLoader'))
        sm.add_widget(DataAnalysis(name='DataAnalysis'))
        sm.add_widget(DataVisualizer(name='DataVisualizer'))
        sm.add_widget(DataML(name='DataML'))
        return sm


if __name__ == '__main__':
    DataLabApp().run()
