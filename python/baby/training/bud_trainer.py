# If you publish results that make use of this software or the Birth Annotator
# for Budding Yeast algorithm, please cite:
# Pietsch, J.M.J., Muñoz, A.F., Adjavon, D.-Y.A., Farquhar, I., Clark, I.B.N.,
# and Swain, P.S. (2023). Determining growth rates from bright-field images of
# budding cells through identifying overlaps. eLife. 12:e79812.
# https://doi.org/10.7554/eLife.79812
# 
# 
# The MIT License (MIT)
# 
# Copyright (c) Julian Pietsch, Alán Muñoz and Diane Adjavon 2023
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.

from pathlib import Path
from itertools import repeat, chain
from warnings import warn
import pickle
import numpy as np
import pandas as pd
from typing import NamedTuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    make_scorer, fbeta_score, accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score
)
import sklearn
if int(sklearn.__version__[0]) > 0:
    from sklearn.metrics import PrecisionRecallDisplay
    plot_precision_recall_curve = PrecisionRecallDisplay.from_estimator
else:
    from sklearn.metrics import plot_precision_recall_curve

from baby.errors import BadProcess, BadParam
from baby.tracker.core import BudTracker

from .utils import (SharedParameterContainer, SharedDataContainer,
                    TrainValTestProperty, TrainValProperty,
                    standard_augmenter)
from .segmentation_trainer import SegmentationTrainer


SCORING_METRICS = {
    'accuracy': make_scorer(accuracy_score),
    'balanced_accuracy': make_scorer(balanced_accuracy_score),
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': make_scorer(f1_score),
    'f0_5': make_scorer(fbeta_score, beta=0.5),
    'f2': make_scorer(fbeta_score, beta=2)
}


class Score(NamedTuple):
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    F1: float
    F0_5: float
    F2: float

    def __str__(self):
        return 'Score({})'.format(', '.join([
            '{}={:.3f}'.format(k, v) for k, v in self._asdict().items()
            ]))


class BudTrainer(BudTracker):
    """Coordinates training for the mother-bud assignment model

    Args:
        kwargs: Additional arguments passed onto the parent Tracker.
        shared_params: Training and segmentation parameters as provided by
            :py:class:`utils.SharedParameterContainer`.
        shared_data: Training data as provided by
            :py:class:`utils.SharedDataContainer`.
        cnn_trainer: Trainer with optimised CNN.
    """

    def __init__(self,
                 shared_params: SharedParameterContainer,
                 shared_data: SharedDataContainer,
                 seg_trainer: SegmentationTrainer,
                 **kwargs):

        kwargs.setdefault('px_size',
                          shared_params.parameters.target_pixel_size)
        kwargs.setdefault('model', False)
        self._shared_params = shared_params
        super().__init__(**kwargs)

        self._shared_data = shared_data
        self._seg_trainer = seg_trainer

        # NB: we inherit self.feats2use from CellTracker class
        self.rf_feats = ["p_bud_mat", "size_ratio_mat", "p_budneck_mat",
                "budneck_ratio_mat", "adjacency_mat"]

        self._model = None

    @property
    def px_size(self):
        return self._shared_params.parameters.target_pixel_size

    @px_size.setter
    def px_size(self, val):
        if val != self._shared_params.parameters.target_pixel_size:
            raise BadParam('px_size should be set via the '
                           '`target_pixel_size` parameter in the '
                           '`SharedParameterContainer`')

    @property
    def save_dir(self):
        """Base directory in which to save trained models"""
        return self._shared_params.save_dir

    @property
    def props_file(self):
        return (self.save_dir /
                self._shared_params.parameters.mother_bud_props_file)

    @property
    def props(self):
        if getattr(self, '_props', None) is None:
            if self.props_file and self.props_file.is_file():
                self.props = pd.read_csv(self.props_file)
            else:
                raise BadProcess(
                        'The property table has not yet been generated')
        return self._props

    @props.setter
    def props(self, props):
        props = pd.DataFrame(props)
        required_cols = self.rf_feats + ['is_mb_pair', 'validation']
        if not all(c in props for c in required_cols):
            raise BadParam(
                '"props" does not have all required columns: {}'.format(
                    ', '.join(required_cols)))
        self._props = props
        props.to_csv(self.props_file)

    @property
    def model_save_file(self):
        return (self.save_dir /
                self._shared_params.parameters.mother_bud_model_file)

    @property
    def model(self):
        if self._model is None:
            if isinstance(getattr(self, '_rf', None), GridSearchCV):
                self._model = self._rf.best_estimator_
            elif self.model_save_file.isfile():
                with open(filename, 'rb') as f:
                    self._model = pickle.load(f)
            else:
                raise BadProcess('"explore_hyperparams" has not been run')
        return self._model

    @model.setter
    def model(self, val):
        if val:
            if not isinstance(val, RandomForestClassifier):
                raise BadParam('model must be a RandomForestClassifier')
            self._model = val
        else:
            self._model = None

    @property
    def feature_importance(self):
        return dict(zip(self.rf_feats, self.model.feature_importances_))

    def generate_property_table(self):
        """Generate table of properties to be used for training
        """
        segtrainer = self._seg_trainer
        flattener = segtrainer.flattener
        data = chain(zip(repeat('train'), segtrainer.examples.train),
                     zip(repeat('val'), segtrainer.examples.val),
                     zip(repeat('test'), segtrainer.examples.test))

        tnames = flattener.names()
        i_budneck = tnames.index('bud_neck')
        bud_target = 'sml_fill' if 'sml_fill' in tnames else 'sml_inte'
        i_bud = tnames.index(bud_target)

        p_list = []
        for train_split, seg_example in data:
            if len(seg_example.target) < 2:
                # Skip if no pairs are present
                continue
            mb_stats = self.calc_mother_bud_stats(seg_example.pred[i_budneck],
                    seg_example.pred[i_bud], seg_example.target)
            p = pd.DataFrame(mb_stats, columns=self.rf_feats)
            p['validation'] = train_split == 'val'
            p['testing'] = train_split == 'test'

            # "cellLabels" specifies the label for each mask
            cell_labels = seg_example.info.get('cellLabels', []) or []
            if type(cell_labels) is int:
                cell_labels = [cell_labels]
            # "buds" specifies the label of the bud for each mask
            buds = seg_example.info.get('buds', []) or []
            if type(buds) is int:
                buds = [buds]

            # Build a ground truth matrix identifying mother-bud pairs
            ncells = len(seg_example.target)
            is_mb_pair = np.zeros((ncells, ncells), dtype=bool)
            mb_inds = [
                (i, cell_labels.index(b))
                for i, b in enumerate(buds)
                if b > 0 and b in cell_labels
            ]
            if len(mb_inds) > 0:
                mother_inds, bud_inds = zip(*mb_inds)
                is_mb_pair[mother_inds, bud_inds] = True
            p['is_mb_pair'] = is_mb_pair.flatten()

            # Ignore any rows containing NaNs
            nanrows = np.isnan(mb_stats).any(axis=1)
            if (p['is_mb_pair'] & nanrows).any():
                id_keys = ('experimentID', 'position', 'trap', 'tp')
                info = seg_example.info
                img_id = ' / '.join(
                        [k + ': ' + str(info[k]) for k in id_keys if k in info])
                warn('Mother-bud pairs omitted due to feature NaNs')
                print('Mother-bud pair omitted in "{}"'.format(img_id))
            p = p.loc[~nanrows, :]
            p_list.append(p)

        props = pd.concat(p_list, ignore_index=True)
        # TODO: should search for any None values in validation column and
        # assign a train-validation split to those rows

        self.props = props # also saves

    def explore_hyperparams(self, hyper_param_target='precision'):
        # Train bud assignment model on validation data, since this more
        # closely represents real-world performance of the CNN:
        data = self.props.loc[self.props['validation'], self.rf_feats]
        truth = self.props.loc[self.props['validation'], 'is_mb_pair']

        rf = RandomForestClassifier(n_estimators=15,
                                    criterion='gini',
                                    max_depth=3,
                                    class_weight='balanced')

        param_grid = {
            'n_estimators': [6, 15, 50, 100],
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': [2, 3, 4],
            'class_weight': [None, 'balanced', 'balanced_subsample']
        }

        def get_balanced_best_index(cv_results_):
            """Find a model balancing F1 score and speed"""
            df = pd.DataFrame(cv_results_)
            best_score = df.iloc[df.mean_test_f1.idxmax(), :]
            thresh = best_score.mean_test_f1 - 0.1 * best_score.std_test_f1
            return df.loc[df.mean_test_f1 > thresh, 'mean_score_time'].idxmin()

        self._model = None
        self._rf = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5,
                scoring=SCORING_METRICS, refit=hyper_param_target)
        self._rf.fit(data, truth)

    def save_model(self, filename=None):
        if filename is None:
            filename = self.model_save_file
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)

    def fit(self, **kwargs):
        try:
            self.props
        except BadProcess:
            self.generate_property_table()
        self.explore_hyperparams(**kwargs)
        self.save_model()

    def performance(self):
        isval = self.props['validation']
        istest = self.props['testing']
        data = self.props.loc[(~isval) & (~istest), self.rf_feats]
        truth = self.props.loc[(~isval) & (~istest), 'is_mb_pair']
        valdata = self.props.loc[isval, self.rf_feats]
        valtruth = self.props.loc[isval, 'is_mb_pair']
        testdata = self.props.loc[istest, self.rf_feats]
        testtruth = self.props.loc[istest, 'is_mb_pair']
        metrics = tuple(SCORING_METRICS.values())
        return TrainValTestProperty(
                Score(*(m(self.model, data, truth) for m in metrics)),
                Score(*(m(self.model, valdata, valtruth) for m in metrics)),
                Score(*(m(self.model, testdata, testtruth) for m in metrics)))

    def grid_search_summary(self):
        if not isinstance(getattr(self, '_rf', None), GridSearchCV):
            raise BadProcess('"explore_hyperparams" has not been run')
        df = pd.DataFrame(self._rf.cv_results_)
        disp_cols = [c for c in df.columns if c.startswith('mean_')
                     or c.startswith('param_')]
        return df.loc[self._rf.best_index_, disp_cols]

    def plot_PR(self):
        if not isinstance(getattr(self, '_rf', None), GridSearchCV):
            raise BadProcess('"explore_hyperparams" has not been run')
        best_rf = self._rf.best_estimator_
        isval = self.props['validation']
        valdata = self.props.loc[isval, self.rf_feats]
        valtruth = self.props.loc[isval, 'is_mb_pair']
        plot_precision_recall_curve(best_rf, valdata, valtruth)

