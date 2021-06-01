#!/usr/bin/env python3
# Author: Vera Bernhard
# Date: 21.06.2021

from numpy.lib.npyio import save
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stress_detector import StressDetector
from stress_detector import FEATURES

# This file describes the steps of finding the best model

# uncomment a steps in the main() function below, to reproduce results of thesis


ALL_FEATURES = [
    'pos', 'pros',
    'nucl_dur', 'syll_dur', 'nucl_dur_norm',  # duration
    'nucl_dur_vnorm', 'syll_dur_norm',  # duration normalised
    'nucl_dur_left', 'nucl_dur_right', 'nucl_dur_v_left',
    'nucl_dur_v_right', 'syll_dur_left', 'syll_dur_right',  # duration context
    'rms', 'int_peak',  # loudness
    'rms_norm', 'int_peak_norm',  # loudness normalised
    'rms_left', 'rms_right',
    'int_peak_left', 'int_peak_right',  # loudness context
    'spect_b1', 'spect_b2', 'spect_b3',  # spectrum
    'spect_b1_left', 'spect_b2_left', 'spect_b3_left', 'spect_b1_right', 'spect_b2_right', 'spect_b3_right',  # spectrum context
    'trajectory', 'f0_max', 'f0_mean', 'f0_meanST', 'f0_max_styl',  # pitch
    'f0_max_norm', 'f0_mean_norm', 'f0_max_styl_norm', 'f0_meanST_norm',  # pitch normalised
    # pitch context
    'intersyllab', 'f0_max_left', 'f0_max_right', 'f0_mean_left', 'f0_mean_right',
    'f0_max_styl_left', 'f0_max_styl_right', 'f0_meanST_left', 'f0_meanST_right'
]

abs_cont = [
    'nucl_dur', 'syll_dur', 'rms', 'int_peak', 'spect_b1', 'spect_b2',
    'spect_b3', 'trajectory', 'f0_max', 'f0_mean', 'f0_meanST', 'f0_max_styl',
    'nucl_dur_left', 'nucl_dur_right', 'nucl_dur_v_left',
    'nucl_dur_v_right', 'syll_dur_left', 'syll_dur_right',
    'rms_left', 'rms_right', 'int_peak_left', 'int_peak_right',
    'spect_b1_left', 'spect_b2_left', 'spect_b3_left', 'spect_b1_right', 'spect_b2_right', 'spect_b3_right',
    'intersyllab', 'f0_max_left', 'f0_max_right', 'f0_mean_left', 'f0_mean_right',
    'f0_max_styl_left', 'f0_max_styl_right', 'f0_meanST_left', 'f0_meanST_right'
]

abs_norm_cont = [
    'nucl_dur', 'syll_dur', 'rms', 'int_peak', 'spect_b1', 'spect_b2',
    'spect_b3', 'trajectory', 'f0_max', 'f0_mean', 'f0_meanST', 'f0_max_styl',
    'nucl_dur_norm',
    'nucl_dur_vnorm', 'syll_dur_norm',
    'rms_norm', 'int_peak_norm',
    'f0_max_norm', 'f0_mean_norm', 'f0_max_styl_norm', 'f0_meanST_norm',
    'nucl_dur_left', 'nucl_dur_right', 'nucl_dur_v_left',
    'nucl_dur_v_right', 'syll_dur_left', 'syll_dur_right',
    'rms_left', 'rms_right', 'int_peak_left', 'int_peak_right',
    'spect_b1_left', 'spect_b2_left', 'spect_b3_left', 'spect_b1_right', 'spect_b2_right', 'spect_b3_right',
    'intersyllab', 'f0_max_left', 'f0_max_right', 'f0_mean_left', 'f0_mean_right',
    'f0_max_styl_left', 'f0_max_styl_right', 'f0_meanST_left', 'f0_meanST_right'
]


wav_path = './wav_tg_all'
par_path = './photrans'


def preprocess():
    """1. Preprocess + get features"""
    sd = StressDetector(wav_path, ALL_FEATURES)
    # Entire preprocess pipeline
    # ----------------------------------------
    sd.preprocess(par_path)
    # alternatively if webmaus and prosogram are already run
    # sd.read_in()

    sd.get_features()
    sd.get_vowel_length('data/vowel_length_test.json')
    sd.get_features().to_csv('./data/complete_features_test.tsv', sep='\t')

    # If preprocess pipeline has already run
    # ----------------------------------------
    # sd.get_features('./data/complete_features.tsv')


def get_best_classifiers():
    """2. Check which 4 algorithms perform best"""
    sd = StressDetector(wav_path, ALL_FEATURES)
    sd.get_features('./data/complete_features.tsv')

    names = [
        "Nearest Neighbors",
        "Logistic Regression",
        "SVM",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "AdaBoost",
        "Naive Bayes",
    ]

    classifiers = [
        KNeighborsClassifier(
            n_jobs=-1
        ),
        LogisticRegression(),
        SVC(probability=True,
            random_state=42),
        DecisionTreeClassifier(
            random_state=42),
        RandomForestClassifier(
            random_state=42,
            n_jobs=-1),
        MLPClassifier(
            random_state=42),
        AdaBoostClassifier(
            random_state=42),
        GaussianNB()]

    # with post-processing
    results_post = (sd.test_classifiers(classifiers, names)).sort_values('f1')

    # without post-processing
    results = sd.test_classifiers(
        classifiers, names, predict_post=False).sort_values('f1')

    print(f"With Post-Processing:\n {results_post}")
    print(f"Without Post-Prossing:\n {results}")

    # ==> Best performing models: Nearest Neighbour, SVM, Random Forest, Neural Net


def select_best_features():
    """3. Select best features/feature groups"""

    sd = StressDetector(wav_path, ALL_FEATURES)
    sd.get_features('./data/complete_features.tsv')

    mlp = MLPClassifier(
        random_state=42,
    )

    nn = KNeighborsClassifier(
        n_jobs=-1,
    )

    svm = SVC(
        random_state=42,
        probability=True,
    )

    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
    )

    classifiers = [mlp, nn, svm, rf]

    names = [
        "Neural Net",
        "Nearest Neighbors",
        "SVM",
        "Random Forest",
    ]

    feat_group1 = {
        'Other Features': ['pos', 'pros'],
        'Duration Features': ['nucl_dur', 'syll_dur', 'nucl_dur_norm',
                              'nucl_dur_vnorm', 'syll_dur_norm',
                              'nucl_dur_left', 'nucl_dur_right', 'nucl_dur_v_left',
                              'nucl_dur_v_right', 'syll_dur_left', 'syll_dur_right'],
        'Loudness Features': ['rms', 'int_peak',
                              'rms_norm', 'int_peak_norm',
                              'rms_left', 'rms_right',
                              'int_peak_left', 'int_peak_right',
                              ],
        'Spectral Features': ['spect_b1', 'spect_b2', 'spect_b3',
                              'spect_b1_left', 'spect_b2_left', 'spect_b3_left', 'spect_b1_right', 'spect_b2_right', 'spect_b3_right'],
        'Pitch Features': ['trajectory', 'f0_max', 'f0_mean', 'f0_meanST', 'f0_max_styl',
                           'f0_max_norm', 'f0_mean_norm', 'f0_max_styl_norm', 'f0_meanST_norm',
                           'intersyllab', 'f0_max_left', 'f0_max_right', 'f0_mean_left', 'f0_mean_right',
                           'f0_max_styl_left', 'f0_max_styl_right', 'f0_meanST_left', 'f0_meanST_right'
                           ]
    }

    feat_group2 = {
        'Absolute': [
            'nucl_dur', 'syll_dur', 'rms', 'int_peak', 'spect_b1', 'spect_b2',
            'spect_b3', 'trajectory', 'f0_max', 'f0_mean', 'f0_meanST', 'f0_max_styl'
        ],
        'Normalized': ['nucl_dur_norm',
                       'nucl_dur_vnorm', 'syll_dur_norm',
                       'rms_norm', 'int_peak_norm',
                       'f0_max_norm', 'f0_mean_norm', 'f0_max_styl_norm', 'f0_meanST_norm',
                       ],
        'Contextual': [
            'nucl_dur_left', 'nucl_dur_right', 'nucl_dur_v_left',
            'nucl_dur_v_right', 'syll_dur_left', 'syll_dur_right',
            'rms_left', 'rms_right', 'int_peak_left', 'int_peak_right',
            'spect_b1_left', 'spect_b2_left', 'spect_b3_left', 'spect_b1_right', 'spect_b2_right', 'spect_b3_right',
            'intersyllab', 'f0_max_left', 'f0_max_right', 'f0_mean_left', 'f0_mean_right',
            'f0_max_styl_left', 'f0_max_styl_right', 'f0_meanST_left', 'f0_meanST_right'
        ],
        'Norm + Cont': ['nucl_dur_norm',
                        'nucl_dur_vnorm', 'syll_dur_norm',
                        'rms_norm', 'int_peak_norm',
                        'f0_max_norm', 'f0_mean_norm', 'f0_max_styl_norm', 'f0_meanST_norm',
                        'nucl_dur_left', 'nucl_dur_right', 'nucl_dur_v_left',
                        'nucl_dur_v_right', 'syll_dur_left', 'syll_dur_right',
                        'rms_left', 'rms_right', 'int_peak_left', 'int_peak_right',
                        'spect_b1_left', 'spect_b2_left', 'spect_b3_left', 'spect_b1_right', 'spect_b2_right', 'spect_b3_right',
                        'intersyllab', 'f0_max_left', 'f0_max_right', 'f0_mean_left', 'f0_mean_right',
                        'f0_max_styl_left', 'f0_max_styl_right', 'f0_meanST_left', 'f0_meanST_right'
                        ],
        'Abs + Cont': ['nucl_dur', 'syll_dur', 'rms', 'int_peak', 'spect_b1', 'spect_b2',
                       'spect_b3', 'trajectory', 'f0_max', 'f0_mean', 'f0_meanST', 'f0_max_styl',
                       'nucl_dur_left', 'nucl_dur_right', 'nucl_dur_v_left',
                       'nucl_dur_v_right', 'syll_dur_left', 'syll_dur_right',
                       'rms_left', 'rms_right', 'int_peak_left', 'int_peak_right',
                       'spect_b1_left', 'spect_b2_left', 'spect_b3_left', 'spect_b1_right', 'spect_b2_right', 'spect_b3_right',
                       'intersyllab', 'f0_max_left', 'f0_max_right', 'f0_mean_left', 'f0_mean_right',
                       'f0_max_styl_left', 'f0_max_styl_right', 'f0_meanST_left', 'f0_meanST_right'
                       ],
        'Abs + Norm + Cont': [
            'nucl_dur', 'syll_dur', 'rms', 'int_peak', 'spect_b1', 'spect_b2',
            'spect_b3', 'trajectory', 'f0_max', 'f0_mean', 'f0_meanST', 'f0_max_styl',
            'nucl_dur_norm',
            'nucl_dur_vnorm', 'syll_dur_norm',
            'rms_norm', 'int_peak_norm',
            'f0_max_norm', 'f0_mean_norm', 'f0_max_styl_norm', 'f0_meanST_norm',
            'nucl_dur_left', 'nucl_dur_right', 'nucl_dur_v_left',
            'nucl_dur_v_right', 'syll_dur_left', 'syll_dur_right',
            'rms_left', 'rms_right', 'int_peak_left', 'int_peak_right',
            'spect_b1_left', 'spect_b2_left', 'spect_b3_left', 'spect_b1_right', 'spect_b2_right', 'spect_b3_right',
            'intersyllab', 'f0_max_left', 'f0_max_right', 'f0_mean_left', 'f0_mean_right',
            'f0_max_styl_left', 'f0_max_styl_right', 'f0_meanST_left', 'f0_meanST_right'
        ]
    }

    for clf, name in zip(classifiers, names):
        outfile_name = f'feature_evaluation/feat_groups1_{name}.tsv'
        outfile_name = outfile_name.replace(' ', '_')
        sd.test_feature_groups(clf, name, feat_group1, outfile_name)

    # # ==> remove 'other' features

    for clf, name in zip(classifiers, names):
        outfile_name = f'feature_evaluation/feat_groups2_{name}.tsv'
        outfile_name = outfile_name.replace(' ', '_')
        sd.test_feature_groups(clf, name, feat_group2, outfile_name)

    # ==> use 'Abs + Cont' and 'Abs + Norm + Cont' for gridsearch

    # try to remove similar or collinear measures manually

    # e.g. removing syllable based measures
    selected_features = [
        'nucl_dur', 'nucl_dur_norm',  # duration
        'nucl_dur_vnorm',   # duration normalised
        'nucl_dur_left', 'nucl_dur_right', 'nucl_dur_v_left',
        'nucl_dur_v_right',  # duration context
        'rms', 'int_peak',  # loudness
        'rms_norm', 'int_peak_norm',  # loudness normalised
        'rms_left', 'rms_right',
        'int_peak_left', 'int_peak_right',  # loudness context
        'spect_b1', 'spect_b2', 'spect_b3',  # spectrum
        'spect_b1_left', 'spect_b2_left', 'spect_b3_left', 'spect_b1_right', 'spect_b2_right', 'spect_b3_right',  # spectrum context
        'trajectory', 'f0_max', 'f0_mean', 'f0_meanST', 'f0_max_styl',  # pitch
        'f0_max_norm', 'f0_mean_norm', 'f0_max_styl_norm', 'f0_meanST_norm',  # pitch normalised
        # pitch context
        'intersyllab', 'f0_max_left', 'f0_max_right', 'f0_mean_left', 'f0_mean_right',
        'f0_max_styl_left', 'f0_max_styl_right', 'f0_meanST_left', 'f0_meanST_right'
    ]

    sd2 = StressDetector(wav_path, selected_features)
    sd2.get_features('./data/complete_features.tsv')

    print(sd2.test_classifiers(classifiers, names))

    # ==> worse result than without removing them, leave all features


def grid_search():
    """4. Grid search on best performing algorithms and best features"""
    pipeline1 = Pipeline([
        ("scaler", StandardScaler()),
        ("nnet", MLPClassifier(
            random_state=42,
            max_iter=300))
    ])

    pipeline2 = Pipeline([
        ("scaler", StandardScaler()),
        ("nn", KNeighborsClassifier(
            n_jobs=-1))

    ])

    pipeline3 = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            probability=True,
            random_state=42)),
    ])

    pipeline4 = Pipeline([
        ("scaler", StandardScaler()),
        ("rf",  RandomForestClassifier(
            random_state=42,
            n_jobs=-1))
    ])

    # 144
    param1 = {
        'nnet__hidden_layer_sizes': [(100,), (50, 50), (100, 50), (50,)],
        'nnet__activation': ['identity', 'logistic', 'tanh', 'relu'],
        'nnet__solver': ['lbfgs', 'sgd', 'adam'],
        'nnet__alpha': [0.001, 0.0001, 0.00001]
    }
    # 64 combinations
    param2 = {
        'nn__n_neighbors': [3, 5, 9, 13],
        'nn__weights': ['uniform', 'distance'],
        'nn__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'nn__metric': ['euclidean', 'manhattan'],
    }

    # 64 combinations
    param3 = {
        'svm__C': [0.01, 0.1, 1.0, 10.0],
        'svm__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'svm__gamma': ['scale', 'auto'],
        'svm__class_weight': ['balanced', None],
    }

    # 144 combinations
    param4 = {
        'rf__n_estimators': [50, 100, 200],
        'rf__criterion': ['gini', 'entropy'],
        'rf__max_depth': [10, 50, 100, None],
        'rf__min_samples_split': [2, 5, 10],
        'rf__class_weight': ['balanced', None]
    }

    pipelines = [pipeline1, pipeline2, pipeline3, pipeline4]
    params = [param1, param2, param3, param4]

    sd = StressDetector(wav_path, abs_cont)
    sd.get_features('./data/complete_features.tsv')
    feat_set1_params = sd.grid_search(pipelines, params)

    sd2 = StressDetector(wav_path, abs_norm_cont)
    sd2.get_features('./data/complete_features.tsv')
    feat_set2_params = sd2.grid_search(pipelines, params)

    print(
        f'Feature Set 1: absolute + context-aware features \n {feat_set1_params}')
    print(
        f'Feature Set 2: absolute + normalized + context-aware features \n {feat_set2_params}')


def retrain_after_gridsearch():
    """5. Retrain models with found hyperparameters"""
    # best parameters for absolute and context-aware features
    mlp_abs_cont = MLPClassifier(
        random_state=42,
        max_iter=300,
        # hyperparameters found by gridsearch
        activation='relu',
        alpha=0.0001,
        hidden_layer_sizes=(100, 50),
        solver='adam'
    )

    nn_abs_cont = KNeighborsClassifier(
        n_jobs=-1,
        # hyperparameters found by gridsearch
        algorithm='auto',
        metric='manhattan',
        n_neighbors=3,
        weights='distance'
    )

    svm_abs_cont = SVC(
        random_state=42,
        probability=True,
        # hyperparameters found by gridsearch
        C=10.0,
        class_weight=None,
        gamma='scale',
        kernel='rbf'
    )

    rf_abs_cont = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        # hyperparameters found by gridsearch
        class_weight='balanced',
        criterion='entropy',
        max_depth=50,
        min_samples_split=5,
        n_estimators=200
    )

    vot_abs_cont = VotingClassifier(
        estimators=[('mlp', mlp_abs_cont), ('nn', nn_abs_cont),
                    ('svm', svm_abs_cont), ('rf', rf_abs_cont)],
        voting='soft')

    # best parameters for absolute, normlised and context-aware features
    mlp_abs_norm_cont = MLPClassifier(
        random_state=42,
        max_iter=300,
        # hyperparameters found by gridsearch
        activation='relu',
        alpha=0.0001,
        hidden_layer_sizes=(100, 50),
        solver='adam'
    )

    nn_abs_norm_cont = KNeighborsClassifier(
        n_jobs=-1,
        # hyperparameters found by gridsearch
        algorithm='auto',
        metric='manhattan',
        n_neighbors=5,
        weights='distance'
    )

    svm_abs_norm_cont = SVC(
        random_state=42,
        probability=True,
        # hyperparameters found by gridsearch
        C=10.0,
        class_weight='balanced',
        gamma='scale',
        kernel='rbf'
    )

    rf_abs_norm_cont = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        # hyperparameters found by gridsearch
        class_weight='balanced',
        criterion='entropy',
        max_depth=50,
        min_samples_split=5,
        n_estimators=100
    )

    vot_abs_norm_cont = VotingClassifier(
        estimators=[('mlp', mlp_abs_norm_cont), ('nn', nn_abs_norm_cont),
                    ('svm', svm_abs_norm_cont), ('rf', rf_abs_norm_cont)],
        voting='soft')

    clf_abs_cont = [mlp_abs_cont, nn_abs_cont,
                    svm_abs_cont, rf_abs_cont, vot_abs_cont]
    clf_abs_norm_cont = [mlp_abs_norm_cont, nn_abs_norm_cont,
                         svm_abs_norm_cont, rf_abs_norm_cont, vot_abs_norm_cont]

    names = [
        "Neural Net",
        "Nearest Neighbors",
        "SVM",
        "Random Forest",
        "Voting"
    ]

    sd = StressDetector(wav_path, abs_cont)
    sd.get_features('./data/complete_features.tsv')
    eval_feat_set1 = sd.test_classifiers(clf_abs_cont, names)

    sd2 = StressDetector(wav_path, abs_norm_cont)
    sd2.get_features('./data/complete_features.tsv')
    eval_feat_set2 = sd2.test_classifiers(clf_abs_norm_cont, names)

    print(
        f'Feature Set 1: absolute + context-aware features \n {eval_feat_set1}')
    print(
        f'Feature Set 2: absolute + normalized + context-aware features \n {eval_feat_set2}')

    # ==> equal performance, choose feature group absolute + context-aware features


def train_best_model():
    """6. Train best system on all data and save it"""

    mlp_abs_cont = MLPClassifier(
        random_state=42,
        max_iter=300,
        # hyperparameters found by gridsearch
        activation='relu',
        alpha=0.0001,
        hidden_layer_sizes=(100, 50),
        solver='adam'
    )

    nn_abs_cont = KNeighborsClassifier(
        n_jobs=-1,
        # hyperparameters found by gridsearch
        algorithm='auto',
        metric='manhattan',
        n_neighbors=3,
        weights='distance'
    )

    svm_abs_cont = SVC(
        random_state=42,
        probability=True,
        # hyperparameters found by gridsearch
        C=10.0,
        class_weight=None,
        gamma='scale',
        kernel='rbf'
    )

    rf_abs_cont = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        # hyperparameters found by gridsearch
        class_weight='balanced',
        criterion='entropy',
        max_depth=50,
        min_samples_split=5,
        n_estimators=200
    )

    vot_abs_cont = VotingClassifier(
        estimators=[('mlp', mlp_abs_cont), ('nn', nn_abs_cont),
                    ('svm', svm_abs_cont), ('rf', rf_abs_cont)],
        voting='soft')

    sd = StressDetector(wav_path, abs_cont)
    sd.get_features('./data/complete_features.tsv')
    sd.train_all(FEATURES, 'vot', save=True)
    print(sd.train(FEATURES, vot_abs_cont, matrix=True))


def load_and_classify():
    """ 7. Make some classification with saved model"""
    sd = StressDetector(wav_path, abs_cont)
    sd.get_features('./data/complete_features.tsv')
    sd.load_classifier('models/classifier_vot_210601.pkl',
                       'models/scaler_vot_210601.pkl')
    sd.classify('test/bamboo1.wav', 'bamboo')
    sd.classify('test/bamboo2.wav', 'bamboo')


def main():

    # 1. Preprocess + get features
    preprocess()

    # 2. Check which 4 algorithms perform best
    # get_best_classifiers()

    # 3. Select best features/feature groups
    # select_best_features()

    # 4. Grid search on best performing algorithms and best features

    # 5. Retrain models with found hyperparameters for both feature set
    # retrain_after_gridsearch()

    # 6. Train best system on all data and save it
    # train_best_model()

    # 7. Make some classification with saved model
    # load_and_classify()


if __name__ == '__main__':
    main()
