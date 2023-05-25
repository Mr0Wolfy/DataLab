# Import base data workflow
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

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
from tkinter import BooleanVar
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
    f1_score, roc_auc_score, mean_squared_error, mean_absolute_error

from sklearn.model_selection import train_test_split

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

        boxlayout.add_widget(button_data_loader)
        boxlayout.add_widget(button_data_analysis)
        boxlayout.add_widget(button_data_visualizer)
        boxlayout.add_widget(button_data_ml)
        floatlayout.add_widget(boxlayout)
        floatlayout.add_widget(button_file_path)
        floatlayout.add_widget(button_upload_data)
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
            text='[color=ff0000]Вставьте название целевой переменной:[/color]',
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
            'Случайный лес',
            'Градиентный бустинг'
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



    def ml_coach(self, *args):

        if 'df' not in globals():
            tk.messagebox.showinfo(title="Ошибка!",
                                   message='Данные для обучения модели не загруженны')
            pass

        elif user_label_input.text == '':
            tk.messagebox.showinfo(title="Ошибка!",
                                   message='Не выбрана целевая переменная')
            pass

        elif 'flag_basic_params' not in globals() and 'flag_optimal_parameters' not in globals():
            tk.messagebox.showinfo(title="Ошибка!",
                                   message='Не заданы параметы модели')
            pass


        else:
            label = user_label_input.text
            X = df.drop(columns=label, axis=1)
            y = df[label]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)
            if flag_basic_params == 1:

                if chosen_model == 'Решающие деревья':
                    if chosen_task == 'Классификация':
                        model = DecisionTreeClassifier()
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        tk.messagebox.showinfo(title="%s" % chosen_model,
                                               message='precision: {}'.format(precision_score(y_test, y_pred)))
                    else:
                        model = DecisionTreeRegressor()
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        tk.messagebox.showinfo(title="%s" % chosen_model,
                                               message='MSE: {}'.format(mean_squared_error(y_test, y_pred)))

                elif chosen_model == 'Логистическая регрессия':
                    model = LogisticRegression()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    tk.messagebox.showinfo(title="%s" % chosen_model,
                                           message='precision: {}'.format(precision_score(y_test, y_pred)))


                elif chosen_model == 'Линейная регрессия':
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    tk.messagebox.showinfo(title="%s" % chosen_model,
                                           message='MSE: {}'.format(mean_squared_error(y_test, y_pred)))

                elif chosen_model == 'Опорные вектора':
                    if chosen_task == 'Классификация':
                        model = LinearSVC()
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        tk.messagebox.showinfo(title="%s" % chosen_model,
                                               message='precision: {}'.format(precision_score(y_test, y_pred)))

                    else:
                        model = LinearSVR()
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        tk.messagebox.showinfo(title="%s" % chosen_model,
                                               message='MSE: {}'.format(mean_squared_error(y_test, y_pred)))

                elif chosen_model == 'Случайный лес':
                    if chosen_task == 'Классификация':
                        model = RandomForestClassifier()
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        tk.messagebox.showinfo(title="%s" % chosen_model,
                                               message='precision: {}'.format(precision_score(y_test, y_pred)))

                    else:
                        model = RandomForestRegressor()
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        tk.messagebox.showinfo(title="%s" % chosen_model,
                                               message='MSE: {}'.format(mean_squared_error(y_test, y_pred)))


                elif chosen_model == 'Ближайшие соседи':
                    if chosen_task == 'Классификация':
                        model = KNeighborsClassifier()
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        tk.messagebox.showinfo(title="%s" % chosen_model,
                                               message='precision: {}'.format(precision_score(y_test, y_pred)))

                    elif chosen_task == 'Регрессия':
                        model = KNeighborsRegressor()
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        tk.messagebox.showinfo(title="%s" % chosen_model,
                                               message='MSE: {}'.format(mean_squared_error(y_test, y_pred)))


    def model_params(self, *args):


        def save_param_tree(*args):
            if chosen_model == 'Решающие деревья':
                global tree_dict
                tree_dict = {}
                if chosen_task == 'Классификация':
                    tree_dict['criterion'] = tree_class_var.get()
                else:
                    tree_dict['criterion'] = tree_reg_var.get()
                tree_dict['max_depth'] = text_input_max_depth.get()
                tree_dict['min_samples_split'] = text_input_min_samples_split.get()
                tk.messagebox.showinfo(title="Устанока параметров модели", message='Параметры установлены!')

            elif chosen_model == 'Логистическая регрессия':
                global log_reg_dict
                log_reg_dict = {}
                log_reg_dict['penalty'] = log_reg_var.get()

            root.destroy()



        def set_basic_params(*args):
            global flag_basic_params
            flag_basic_params = 1
            tk.messagebox.showinfo(title="Устанока параметров модели", message='Параметры установлены!')
            root.destroy()

        def set_optimal_parameters(*args):
            global flag_optimal_parameters
            flag_optimal_parameters = 1
            tk.messagebox.showinfo(title="Устанока параметров модели", message='Параметры будут подобраны автоматически!')
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

                button_save_param = tk.Button(root, text="Сохранить параметры", command=save_param_tree)
                button_save_param.pack()
                button_save_param.place(x=20, y=200)

                button_basic_param = tk.Button(root, text="Параметры по умолчанию", command=set_basic_params)
                button_basic_param.pack()
                button_basic_param.place(x=20, y=250)

                button_auto_param = tk.Button(root, text="Автоподбор параметров")
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

                    root.mainloop()

                elif chosen_model == 'Опорные вектора':
                    root.mainloop()

                elif chosen_model == 'Ближайшие соседи':
                    if chosen_task == 'Классификация':
                        root.mainloop()
                    elif chosen_task == 'Регрессия':
                        root.mainloop()
                    else:
                        root.mainloop()


                elif chosen_model == 'Случайный лес':

                    label_criterion = tk.Label(text='Выберите критейрий: ')
                    label_criterion.pack()
                    label_criterion.place(x=20, y=20)

                    label_n_estimators = tk.Label(text='Количество деревьев: ')
                    label_n_estimators.pack()
                    label_n_estimators.place(x=20, y=50)

                    text_input_n_estimators = ttk.Entry(width=10)
                    text_input_n_estimators.pack()
                    text_input_n_estimators.place(x=190, y=50)

                    label_max_depth = tk.Label(text='Максимальная глубина деревьев: ')
                    label_max_depth.pack()
                    label_max_depth.place(x=20, y=80)

                    text_input_max_depth = ttk.Entry()
                    text_input_max_depth.pack()
                    text_input_max_depth.place(x=270, y=80)

                    label_min_samples_leaf = tk.Label(text='Минимальное кол-во элементов в листе: ')
                    label_min_samples_leaf.pack()
                    label_min_samples_leaf.place(x=20, y=110)

                    text_input_min_samples_leaf = ttk.Entry()
                    text_input_min_samples_leaf.pack()
                    text_input_min_samples_leaf.place(x=315, y=110)

                    label_min_samples_split = tk.Label(text='Минимальное кол-во элементов для разделения: ')
                    label_min_samples_split.pack()
                    label_min_samples_split.place(x=20, y=140)

                    text_input_min_samples_split = ttk.Entry()
                    text_input_min_samples_split.pack()
                    text_input_min_samples_split.place(x=375, y=140)

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
